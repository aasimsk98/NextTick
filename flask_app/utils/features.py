"""
Feature engineering for NextTick.

IMPORTANT: the logic in this module must stay byte-for-byte aligned with the
`engineer_features` function in notebooks/NextTick_Training.ipynb. If the
training notebook's features change, update this file in lock-step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS: list[str] = [
    "Return_1d", "LogReturn_1d",
    "SMA_5", "SMA_10", "SMA_20",
    "Close_over_SMA_5", "Close_over_SMA_10", "Close_over_SMA_20",
    "Volatility_5", "Volatility_10", "Volatility_20",
    "Momentum_3", "Momentum_5", "Momentum_10",
    "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Position",
    "Volume_Ratio",
    "HL_Range", "Close_Position",
]


REQUIRED_INPUT_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive predictive indicators from OHLCV data.

    Parameters
    ----------
    df : DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``, ``Volume``
        sorted in ascending chronological order.
    """
    out = df.copy()

    # Returns
    out["Return_1d"] = out["Close"].pct_change()
    out["LogReturn_1d"] = np.log(out["Close"] / out["Close"].shift(1))

    # Moving averages and close/SMA ratios
    for w in (5, 10, 20):
        out[f"SMA_{w}"] = out["Close"].rolling(w).mean()
        out[f"Close_over_SMA_{w}"] = out["Close"] / out[f"SMA_{w}"] - 1

    # Volatility (rolling std of returns)
    for w in (5, 10, 20):
        out[f"Volatility_{w}"] = out["Return_1d"].rolling(w).std()

    # Momentum (N-day percentage change)
    for w in (3, 5, 10):
        out[f"Momentum_{w}"] = out["Close"] / out["Close"].shift(w) - 1

    # Relative Strength Index
    out["RSI_14"] = _rsi(out["Close"], 14)

    # MACD family
    ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_12 - ema_26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    # Bollinger position (z-score within 2σ band)
    bb_mid = out["Close"].rolling(20).mean()
    bb_std = out["Close"].rolling(20).std()
    out["BB_Position"] = (out["Close"] - bb_mid) / (2 * bb_std)

    # Volume ratio
    out["Volume_Ratio"] = out["Volume"] / out["Volume"].rolling(20).mean()

    # Intraday range / position
    out["HL_Range"] = (out["High"] - out["Low"]) / out["Close"]
    out["Close_Position"] = (out["Close"] - out["Low"]) / (
        out["High"] - out["Low"]
    ).replace(0, np.nan)

    return out


def validate_input_frame(df: pd.DataFrame, min_rows: int = 30) -> None:
    """Raise ``ValueError`` if the input frame is unsuitable for inference."""
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected: {REQUIRED_INPUT_COLUMNS}."
        )
    if len(df) < min_rows:
        raise ValueError(
            f"At least {min_rows} rows of OHLCV data are required "
            f"(received {len(df)})."
        )
    # Numeric sanity
    for col in REQUIRED_INPUT_COLUMNS:
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"Column '{col}' must be numeric.")
