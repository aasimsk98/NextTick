"""
Model loading and inference for NextTick.

Loads six trained models + the fitted scaler, then exposes a single
``predict(df, market_df, ticker)`` entry point used by the Flask view layer.

Models in this build:
  - Logistic Regression / Random Forest Classifier / LSTM Classifier (PyTorch)
  - Linear Regression / Random Forest Regressor / LSTM Regressor (PyTorch)

Scaler and sklearn models are pickled. LSTMs are saved as whole-model
``torch.save`` files (.pt).
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.features import (
    FEATURE_COLUMNS,
    engineer_features,
    validate_input_frame,
)

logger = logging.getLogger(__name__)

# Lazy-import PyTorch so the Flask app can still boot (and display a friendly
# error page) when TensorFlow-era artifacts are present and PyTorch is not.
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception as exc:
    logger.warning("PyTorch unavailable - LSTM predictions disabled: %s", exc)
    _HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore


# PyTorch's ``torch.load`` on a whole-model file requires the model class to be
# importable at load time. We redefine the exact classes used at training
# time (Phase 2 and Phase 3 notebooks) so the pickled graphs can be rebuilt.
if _HAS_TORCH:

    class LSTMClassifier(nn.Module):
        """Stacked LSTM classifier: LSTM(64) -> LSTM(32) -> Dense(16) -> Dense(1).

        Output is a raw logit; apply sigmoid at inference to get P(Up).
        """
        def __init__(self, n_features, hidden_1=64, hidden_2=32,
                     dense_hidden=16, dropout=0.2):
            super().__init__()
            self.lstm1 = nn.LSTM(n_features, hidden_1, batch_first=True)
            self.dropout1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(hidden_1, hidden_2, batch_first=True)
            self.dropout2 = nn.Dropout(dropout)
            self.dense1 = nn.Linear(hidden_2, dense_hidden)
            self.relu = nn.ReLU()
            self.output = nn.Linear(dense_hidden, 1)

        def forward(self, x):
            lstm1_out, _ = self.lstm1(x)
            lstm1_out = self.dropout1(lstm1_out)
            _, (h_n, _) = self.lstm2(lstm1_out)
            final_hidden = h_n[-1]
            final_hidden = self.dropout2(final_hidden)
            h = self.relu(self.dense1(final_hidden))
            return self.output(h).squeeze(-1)


    class LSTMRegressor(nn.Module):
        """Same architecture as LSTMClassifier; linear output for regression."""
        def __init__(self, n_features, hidden_1=64, hidden_2=32,
                     dense_hidden=16, dropout=0.2):
            super().__init__()
            self.lstm1 = nn.LSTM(n_features, hidden_1, batch_first=True)
            self.dropout1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(hidden_1, hidden_2, batch_first=True)
            self.dropout2 = nn.Dropout(dropout)
            self.dense1 = nn.Linear(hidden_2, dense_hidden)
            self.relu = nn.ReLU()
            self.output = nn.Linear(dense_hidden, 1)

        def forward(self, x):
            lstm1_out, _ = self.lstm1(x)
            lstm1_out = self.dropout1(lstm1_out)
            _, (h_n, _) = self.lstm2(lstm1_out)
            final_hidden = h_n[-1]
            final_hidden = self.dropout2(final_hidden)
            h = self.relu(self.dense1(final_hidden))
            return self.output(h).squeeze(-1)


# Data classes

@dataclass
class ModelPrediction:
    """Single model's output for one of the two tasks."""
    model: str
    task: str                    # "classification" | "regression"
    value: float                 # probability of "Up"  OR  predicted % change (in percent points)
    label: Optional[str] = None  # "Up" / "Down"  for classification


@dataclass
class InferenceResult:
    """Bundle of all model outputs + an ensemble summary."""
    classifications: list[ModelPrediction] = field(default_factory=list)
    regressions:     list[ModelPrediction] = field(default_factory=list)

    # Ensemble summary
    direction:           Optional[str]   = None  # "Up" / "Down"
    direction_confidence: Optional[float] = None # 0..1
    magnitude_pct:       Optional[float] = None  # average predicted % change (percent points)

    # Context data echoed back to the UI
    last_close:  Optional[float] = None
    next_close:  Optional[float] = None          # projected from magnitude
    history:     list[dict]      = field(default_factory=list)

    # Detailed walkthrough data
    data_summary:      dict       = field(default_factory=dict)
    features_snapshot: list[dict] = field(default_factory=list)
    ensemble_detail:   dict       = field(default_factory=dict)


# Human-readable metadata for the 21 features, shown in the UI walkthrough.
# Keys match FEATURE_COLUMNS exactly.
_FEAT_META: list[tuple[str, str, str]] = [
    ("rsi_14",             "RSI (14)",            "Relative Strength Index - momentum oscillator. >70 = overbought, <30 = oversold."),
    ("sma_10",             "SMA 10",              "10-day simple moving average of close price - short-term trend."),
    ("sma_20",             "SMA 20",              "20-day simple moving average of close - medium-term trend benchmark."),
    ("momentum_10",        "Momentum (10d)",      "10-day price change: Close / Close[-10] - 1."),
    ("volatility_10",      "Volatility (10d)",    "10-day rolling standard deviation of daily returns."),
    ("daily_return",       "Daily Return",        "Simple daily return: previous session's percentage change."),
    ("spy_return",         "SPY Return",          "Daily return of the broad US equity market (SPY ETF)."),
    ("vix_level",          "VIX Level",           "CBOE volatility index - market 'fear gauge'."),
    ("sector_return",      "Sector Return",       "Daily return of this stock's sector ETF (XLK, XLF, XLV, etc.)."),
    ("relative_to_spy",    "Relative to SPY",     "Stock return minus SPY return - isolates stock-specific move."),
    ("relative_to_sector", "Relative to Sector",  "Stock return minus its sector ETF return."),
    ("tnx_change",         "10Y Yield Change",    "Daily change in the 10-year US Treasury yield."),
    ("dxy_change",         "Dollar Index Change", "Daily percent change in the US Dollar index (DXY)."),
    ("oil_return",         "Oil Return",          "Daily return of USO ETF (oil price proxy)."),
    ("overnight_gap",      "Overnight Gap",       "Today's open relative to yesterday's close, in percent."),
    ("intraday_return",    "Intraday Return",     "Close vs open within the same session."),
    ("daily_range_pct",    "H-L Range / Close",   "Intraday range as % of close - a measure of session volatility."),
    ("close_location",     "Close Position",      "Where close sits within the day's High-Low range (0 = Low, 1 = High)."),
    ("relative_volume",    "Volume Ratio",        "Today's volume divided by its 20-day average. >1 = elevated activity."),
    ("day_of_week",        "Day of Week",         "Calendar feature: 0=Monday ... 4=Friday."),
    ("month",              "Month",               "Calendar feature: 1 ... 12."),
]


class InferenceService:
    """Loads models from ``models_dir`` once and serves predictions."""

    # Filenames produced by the Phase 2 and Phase 3 notebooks.
    FILES = {
        "logreg":   "logistic_regression.pkl",
        "rf_cls":   "random_forest_classifier.pkl",
        "lin_reg":  "linear_regression.pkl",
        "rf_reg":   "random_forest_regressor.pkl",
        "lstm_cls": "lstm_classifier.pt",
        "lstm_reg": "lstm_regressor.pt",
        "scaler":   "scaler.pkl",
        "metadata": "metadata.json",
    }

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.metadata: dict = {}
        self.feature_columns: list[str] = FEATURE_COLUMNS
        # Training notebooks used a 30-day window for the LSTMs
        self.lstm_window: int = 30
        self.min_rows_required: int = 40  # 30-day window + a few rows of warmup buffer

        self.scaler     = None
        self.logreg     = None
        self.rf_cls     = None
        self.lin_reg    = None
        self.rf_reg     = None
        self.lstm_cls   = None
        self.lstm_reg   = None

        self.status: dict[str, str] = {}
        self._load()

    # internals

    def _path(self, key: str) -> str:
        return os.path.join(self.models_dir, self.FILES[key])

    def _safe_pickle(self, key: str):
        """Load a pickled sklearn / scaler artifact."""
        p = self._path(key)
        if not os.path.exists(p):
            self.status[key] = "missing"
            logger.warning("Artifact '%s' not found at %s", key, p)
            return None
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            self.status[key] = "ok"
            return obj
        except Exception as exc:
            self.status[key] = f"error: {exc}"
            logger.exception("Failed to load %s", p)
            return None

    def _safe_torch(self, key: str):
        """Load a whole-model torch.save artifact."""
        if not _HAS_TORCH:
            self.status[key] = "torch-unavailable"
            return None
        p = self._path(key)
        if not os.path.exists(p):
            self.status[key] = "missing"
            return None
        try:
            # weights_only=False required for whole-model loads on torch>=2.6.
            # The model classes (LSTMClassifier / LSTMRegressor) are defined
            # in this module so pickle can find them.
            obj = torch.load(p, map_location="cpu", weights_only=False)
            obj.eval()
            self.status[key] = "ok"
            return obj
        except Exception as exc:
            self.status[key] = f"error: {exc}"
            logger.exception("Failed to load PyTorch model %s", p)
            return None

    def _load(self) -> None:
        meta_path = self._path("metadata")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.feature_columns    = self.metadata.get("feature_columns", FEATURE_COLUMNS)
            self.lstm_window        = int(self.metadata.get("lstm_window", 30))
            self.min_rows_required  = int(self.metadata.get("min_rows_required", 40))
            self.status["metadata"] = "ok"
        else:
            # Metadata is optional - we have sensible defaults above.
            self.status["metadata"] = "missing-using-defaults"

        self.scaler   = self._safe_pickle("scaler")
        self.logreg   = self._safe_pickle("logreg")
        self.rf_cls   = self._safe_pickle("rf_cls")
        self.lin_reg  = self._safe_pickle("lin_reg")
        self.rf_reg   = self._safe_pickle("rf_reg")
        self.lstm_cls = self._safe_torch("lstm_cls")
        self.lstm_reg = self._safe_torch("lstm_reg")

    # public

    @property
    def is_ready(self) -> bool:
        """True iff the minimum set of artifacts for a prediction is loaded."""
        return (
            self.scaler is not None
            and self.logreg is not None
            and self.rf_cls is not None
            and self.lin_reg is not None
            and self.rf_reg is not None
        )

    def predict(
        self,
        df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None,
        ticker: Optional[str] = None,
    ) -> InferenceResult:
        """Run all available models on a 30+ day OHLCV frame.

        Parameters
        ----------
        df : DataFrame
            User-ticker OHLCV indexed by date.
        market_df : DataFrame, optional
            Market context from ``fetcher.fetch_market_context``. Needed to
            compute the 10 market/macro features.
        ticker : str, optional
            Used to choose the correct sector ETF.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Models are not loaded. Place the artifacts produced by the "
                "training notebooks into the `models/` directory and restart."
            )

        validate_input_frame(df, min_rows=self.min_rows_required)

        # The fetcher returns a reset-index DataFrame with a 'Date' column.
        # Feature engineering expects a DatetimeIndex, so set it back if needed.
        df_sorted = df.copy()
        if "Date" in df_sorted.columns:
            df_sorted = df_sorted.set_index("Date")
        df_sorted = df_sorted.sort_index()

        last_close = float(df_sorted["Close"].iloc[-1])

        # Engineer the 21 features using market context
        feat_df = engineer_features(df_sorted, market_df=market_df, ticker=ticker).dropna()
        if feat_df.empty:
            raise ValueError(
                "Feature engineering produced zero usable rows - "
                "the input data likely contains NaNs or too few rows."
            )

        # Scale the most recent row for the non-sequential models
        X_last = self.scaler.transform(
            feat_df[self.feature_columns].iloc[[-1]].values
        )

        # Features snapshot for the walkthrough UI
        last_feat = feat_df.iloc[-1]
        features_snapshot = []
        for key, label, desc in _FEAT_META:
            try:
                raw = float(last_feat[key])
                val = None if np.isnan(raw) else round(raw, 4)
            except (KeyError, ValueError, TypeError):
                val = None
            features_snapshot.append({"key": key, "label": label, "desc": desc, "value": val})

        result = InferenceResult(last_close=last_close)

        result.data_summary      = {
            "rows":        len(df_sorted),
            "date_from":   str(pd.Timestamp(df_sorted.index[0]).date()),
            "date_to":     str(pd.Timestamp(df_sorted.index[-1]).date()),
            "last_open":   round(float(df_sorted["Open"].iloc[-1]),   2),
            "last_high":   round(float(df_sorted["High"].iloc[-1]),   2),
            "last_low":    round(float(df_sorted["Low"].iloc[-1]),    2),
            "last_close":  round(float(df_sorted["Close"].iloc[-1]),  2),
            "last_volume": int(df_sorted["Volume"].iloc[-1]),
        }
        result.features_snapshot = features_snapshot

        # Direction (classification)
        p_logreg = float(self.logreg.predict_proba(X_last)[0, 1])
        p_rf_cls = float(self.rf_cls.predict_proba(X_last)[0, 1])

        result.classifications.append(ModelPrediction(
            model="Logistic Regression",
            task="classification",
            value=p_logreg,
            label="Up" if p_logreg >= 0.5 else "Down",
        ))
        result.classifications.append(ModelPrediction(
            model="Random Forest Classifier",
            task="classification",
            value=p_rf_cls,
            label="Up" if p_rf_cls >= 0.5 else "Down",
        ))

        # Magnitude (regression)
        # Our regressors were trained on ``target_return`` which is a raw fraction
        # (e.g. 0.015 for +1.5%). The Flask UI expects magnitude in percent
        # points so it can compute ``next_close = last_close * (1 + mag / 100)``.
        # We multiply by 100 here so the rest of the pipeline sees percent points.
        v_lin = float(self.lin_reg.predict(X_last)[0]) * 100.0
        v_rf  = float(self.rf_reg.predict(X_last)[0])  * 100.0

        result.regressions.append(ModelPrediction(
            model="Linear Regression",
            task="regression",
            value=v_lin,
        ))
        result.regressions.append(ModelPrediction(
            model="Random Forest Regressor",
            task="regression",
            value=v_rf,
        ))

        # LSTM (optional - only if enough rows and both torch models loaded)
        if (_HAS_TORCH
                and len(feat_df) >= self.lstm_window
                and self.lstm_cls is not None
                and self.lstm_reg is not None):
            # Build a (1, lstm_window, n_features) sequence scaled with the
            # same fitted scaler used for the sklearn models.
            seq_np = self.scaler.transform(
                feat_df[self.feature_columns].tail(self.lstm_window).values
            ).reshape(1, self.lstm_window, -1).astype(np.float32)

            with torch.no_grad():
                seq_t = torch.from_numpy(seq_np)

                # Classifier outputs a logit - apply sigmoid to get P(Up)
                logit = self.lstm_cls(seq_t).item()
                p_lstm = 1.0 / (1.0 + float(np.exp(-logit)))

                # Regressor outputs raw fractional return - scale to percent points
                v_lstm = self.lstm_reg(seq_t).item() * 100.0

            result.classifications.append(ModelPrediction(
                model="LSTM Classifier",
                task="classification",
                value=p_lstm,
                label="Up" if p_lstm >= 0.5 else "Down",
            ))
            result.regressions.append(ModelPrediction(
                model="LSTM Regressor",
                task="regression",
                value=v_lstm,
            ))

        # Ensemble summary
        probs = [p.value for p in result.classifications]
        mags  = [p.value for p in result.regressions]

        avg_prob = float(np.mean(probs))
        avg_mag  = float(np.mean(mags))

        result.direction             = "Up" if avg_prob >= 0.5 else "Down"
        # Confidence = distance from 0.5, scaled to 0..1
        result.direction_confidence  = float(abs(avg_prob - 0.5) * 2)
        result.magnitude_pct         = avg_mag
        result.next_close            = last_close * (1 + avg_mag / 100.0)

        votes_up   = sum(1 for p in result.classifications if p.value >= 0.5)
        votes_down = len(result.classifications) - votes_up
        result.ensemble_detail = {
            "votes_up":           votes_up,
            "votes_down":         votes_down,
            "total_classifiers":  len(result.classifications),
            "total_regressors":   len(result.regressions),
            "avg_prob":           round(avg_prob, 4),
            "avg_mag":            round(avg_mag, 4),
            "confidence_formula": f"|{avg_prob:.4f} - 0.5| * 2 = {result.direction_confidence:.4f}",
            "projection_formula": f"${last_close:.2f} * (1 + {avg_mag:.4f} / 100) = ${result.next_close:.2f}",
        }

        # History snapshot for charting (last 30 rows)
        hist = df_sorted.tail(30).reset_index()
        date_col = hist.columns[0]
        result.history = [
            {
                "date": (row[date_col].strftime("%Y-%m-%d")
                         if hasattr(row[date_col], "strftime")
                         else str(row[date_col])),
                "close": float(row["Close"]),
            }
            for _, row in hist.iterrows()
        ]

        return result
