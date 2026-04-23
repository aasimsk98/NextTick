"""
Model loading and inference for NextTick.

Loads six trained models + the fitted scaler, then exposes a single
``predict(df)`` entry point used by the Flask view layer.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from utils.features import (
    FEATURE_COLUMNS,
    engineer_features,
    validate_input_frame,
)

logger = logging.getLogger(__name__)

# Lazy-import TensorFlow so the Flask app can still start (and display a
# friendly error page) when only sklearn artifacts are present.
try:
    from tensorflow.keras.models import load_model as keras_load_model  # type: ignore

    _HAS_TF = True
except Exception as exc:  # noqa: BLE001
    logger.warning("TensorFlow unavailable — LSTM predictions disabled: %s", exc)
    _HAS_TF = False
    keras_load_model = None  # type: ignore


# -------------------------------------------------------------- #
# Data classes
# -------------------------------------------------------------- #
@dataclass
class ModelPrediction:
    """Single model's output for one of the two tasks."""
    model: str
    task: str                    # "classification" | "regression"
    value: float                 # probability of "Up"  OR  predicted % change
    label: Optional[str] = None  # "Up" / "Down"  for classification


@dataclass
class InferenceResult:
    """Bundle of all model outputs + an ensemble summary."""
    classifications: list[ModelPrediction] = field(default_factory=list)
    regressions:     list[ModelPrediction] = field(default_factory=list)

    # Ensemble summary
    direction:           Optional[str]   = None  # "Up" / "Down"
    direction_confidence: Optional[float] = None # 0..1
    magnitude_pct:       Optional[float] = None  # average predicted % change

    # Context data echoed back to the UI
    last_close:  Optional[float] = None
    next_close:  Optional[float] = None          # projected from magnitude
    history:     list[dict]      = field(default_factory=list)

    # Detailed walkthrough data
    data_summary:      dict       = field(default_factory=dict)
    features_snapshot: list[dict] = field(default_factory=list)
    ensemble_detail:   dict       = field(default_factory=dict)


# Human-readable metadata for each engineered feature shown in the UI walkthrough.
_FEAT_META: list[tuple[str, str, str]] = [
    ("RSI_14",            "RSI (14)",            "Relative Strength Index — momentum oscillator. >70 = overbought, <30 = oversold."),
    ("MACD",              "MACD",                "EMA(12) − EMA(26). Positive = bullish momentum building."),
    ("MACD_Signal",       "MACD Signal",         "9-day EMA of MACD. When MACD crosses above this line it is a buy signal."),
    ("MACD_Hist",         "MACD Histogram",      "MACD − Signal. Positive and rising = strengthening uptrend."),
    ("BB_Position",       "Bollinger Position",  "Z-score within 2σ Bollinger Bands. >0 = above midline, <0 = below."),
    ("SMA_5",             "SMA 5",               "5-day simple moving average of close price — short-term trend."),
    ("SMA_20",            "SMA 20",              "20-day simple moving average — medium-term trend benchmark."),
    ("Close_over_SMA_5",  "Close / SMA5 − 1",   "% deviation of today's close from the 5-day average."),
    ("Close_over_SMA_20", "Close / SMA20 − 1",  "% deviation of today's close from the 20-day average."),
    ("Momentum_5",        "Momentum (5d)",       "5-day price change: Close / Close[−5] − 1."),
    ("Momentum_10",       "Momentum (10d)",      "10-day price change: Close / Close[−10] − 1."),
    ("Volatility_5",      "Volatility (5d)",     "5-day rolling standard deviation of daily returns."),
    ("Volatility_20",     "Volatility (20d)",    "20-day rolling standard deviation of daily returns."),
    ("Volume_Ratio",      "Volume Ratio",        "Today's volume divided by its 20-day average. >1 = elevated activity."),
    ("Return_1d",         "1-day Return",        "Previous session's simple daily return (pct_change)."),
    ("HL_Range",          "H-L Range / Close",   "Intraday range as % of close — a measure of session volatility."),
    ("Close_Position",    "Close Position",      "Where close sits within the day's High-Low range (0 = Low, 1 = High)."),
]


# -------------------------------------------------------------- #
# Loader
# -------------------------------------------------------------- #
class InferenceService:
    """Loads models from ``models_dir`` once and serves predictions."""

    # Filenames produced by the training notebook
    FILES = {
        "logreg":   "logistic_regression.pkl",
        "rf_cls":   "random_forest_classifier.pkl",
        "lin_reg":  "linear_regression.pkl",
        "rf_reg":   "random_forest_regressor.pkl",
        "lstm_cls": "lstm_classifier.keras",
        "lstm_reg": "lstm_regressor.keras",
        "scaler":   "scaler.pkl",
        "metadata": "metadata.json",
    }

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.metadata: dict = {}
        self.feature_columns: list[str] = FEATURE_COLUMNS
        self.lstm_window: int = 20
        self.min_rows_required: int = 30

        self.scaler     = None
        self.logreg     = None
        self.rf_cls     = None
        self.lin_reg    = None
        self.rf_reg     = None
        self.lstm_cls   = None
        self.lstm_reg   = None

        self.status: dict[str, str] = {}
        self._load()

    # ---- internals ------------------------------------------- #
    def _path(self, key: str) -> str:
        return os.path.join(self.models_dir, self.FILES[key])

    def _safe_joblib(self, key: str):
        p = self._path(key)
        if not os.path.exists(p):
            self.status[key] = "missing"
            logger.warning("Artifact '%s' not found at %s", key, p)
            return None
        try:
            obj = joblib.load(p)
            self.status[key] = "ok"
            return obj
        except Exception as exc:  # noqa: BLE001
            self.status[key] = f"error: {exc}"
            logger.exception("Failed to load %s", p)
            return None

    def _safe_keras(self, key: str):
        if not _HAS_TF:
            self.status[key] = "tensorflow-unavailable"
            return None
        p = self._path(key)
        if not os.path.exists(p):
            self.status[key] = "missing"
            return None
        try:
            obj = keras_load_model(p, compile=False)
            self.status[key] = "ok"
            return obj
        except Exception as exc:  # noqa: BLE001
            self.status[key] = f"error: {exc}"
            logger.exception("Failed to load Keras model %s", p)
            return None

    def _load(self) -> None:
        meta_path = self._path("metadata")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.feature_columns    = self.metadata.get("feature_columns", FEATURE_COLUMNS)
            self.lstm_window        = int(self.metadata.get("lstm_window", 20))
            self.min_rows_required  = int(self.metadata.get("min_rows_required", 30))
            self.status["metadata"] = "ok"
        else:
            self.status["metadata"] = "missing"

        self.scaler   = self._safe_joblib("scaler")
        self.logreg   = self._safe_joblib("logreg")
        self.rf_cls   = self._safe_joblib("rf_cls")
        self.lin_reg  = self._safe_joblib("lin_reg")
        self.rf_reg   = self._safe_joblib("rf_reg")
        self.lstm_cls = self._safe_keras("lstm_cls")
        self.lstm_reg = self._safe_keras("lstm_reg")

    # ---- public ---------------------------------------------- #
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

    def predict(self, df: pd.DataFrame) -> InferenceResult:
        """Run all available models on a 30+ day OHLCV frame."""
        if not self.is_ready:
            raise RuntimeError(
                "Models are not loaded. Place the artifacts produced by the "
                "training notebook into the `models/` directory and restart."
            )

        validate_input_frame(df, min_rows=self.min_rows_required)

        # Preserve the original close so we can project tomorrow's price
        df_sorted = df.sort_index()
        last_close = float(df_sorted["Close"].iloc[-1])

        feat_df = engineer_features(df_sorted).dropna()
        if feat_df.empty:
            raise ValueError(
                "Feature engineering produced zero usable rows — "
                "the input data likely contains NaNs or too few rows."
            )

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

        # Data summary for the walkthrough UI
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

        # ---- Direction (classification) ---------------------- #
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

        # ---- Magnitude (regression) -------------------------- #
        v_lin = float(self.lin_reg.predict(X_last)[0])
        v_rf  = float(self.rf_reg.predict(X_last)[0])

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

        # ---- LSTM (optional) --------------------------------- #
        if len(feat_df) >= self.lstm_window and self.lstm_cls is not None and self.lstm_reg is not None:
            seq = self.scaler.transform(
                feat_df[self.feature_columns].tail(self.lstm_window).values
            ).reshape(1, self.lstm_window, -1).astype(np.float32)

            p_lstm = float(self.lstm_cls.predict(seq, verbose=0).flatten()[0])
            v_lstm = float(self.lstm_reg.predict(seq, verbose=0).flatten()[0])

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

        # ---- Ensemble summary -------------------------------- #
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
            "confidence_formula": f"|{avg_prob:.4f} − 0.5| × 2 = {result.direction_confidence:.4f}",
            "projection_formula": f"${last_close:.2f} × (1 + {avg_mag:.4f} / 100) = ${result.next_close:.2f}",
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
