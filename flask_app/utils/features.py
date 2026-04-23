"""
Feature engineering for NextTick.

IMPORTANT: the logic in this module must stay byte-for-byte aligned with the
feature engineering in notebooks/01_data_pipeline.ipynb. If the training
notebook's features change, update this file in lock-step.

This module produces the exact 21 features the models were trained on:
  - 6 price/technical features from the user's ticker
  - 10 market/macro features from market instruments (SPY, VIX, TNX, DXY, USO,
    and the user-ticker's sector ETF)
  - 5 OHLCV-derived features from the user's ticker
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# The exact 21 features the models expect, in the exact column order used at
# training time. DO NOT reorder this list - the fitted scaler was fit on this
# ordering, so inference must preserve it.
FEATURE_COLUMNS: list[str] = [
    # Price / technical (6)
    "daily_return", "sma_10", "sma_20", "volatility_10", "momentum_10", "rsi_14",
    # Market / macro (10)
    "spy_return", "vix_level", "sector_return", "relative_to_spy", "relative_to_sector",
    "tnx_change", "dxy_change", "oil_return", "day_of_week", "month",
    # OHLCV-derived (5)
    "overnight_gap", "intraday_return", "daily_range_pct", "close_location", "relative_volume",
]

# Minimum input columns on the user's ticker frame
REQUIRED_INPUT_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]

# Ticker -> sector ETF mapping. Covers the 50 stocks the models were trained on.
# If the user types a ticker outside this list, we fall back to SPY as the
# sector benchmark (see engineer_features).
TICKER_TO_SECTOR_ETF: dict[str, str] = {
    # Technology (XLK)
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "NVDA": "XLK",
    "CRM":  "XLK", "ORCL": "XLK", "ADBE": "XLK",
    # Consumer Discretionary (XLY)
    "AMZN": "XLY", "TSLA": "XLY", "HD":   "XLY", "NKE":  "XLY",
    "MCD":  "XLY", "SBUX": "XLY",
    # Financials (XLF)
    "JPM":  "XLF", "GS":   "XLF", "BAC":  "XLF", "V":    "XLF",
    "MA":   "XLF", "BLK":  "XLF",
    # Healthcare (XLV)
    "JNJ":  "XLV", "PFE":  "XLV", "UNH":  "XLV", "LLY":  "XLV",
    "ABBV": "XLV", "MRK":  "XLV", "TMO":  "XLV",
    # Energy (XLE)
    "XOM":  "XLE", "CVX":  "XLE", "COP":  "XLE", "SLB":  "XLE",
    "EOG":  "XLE",
    # Industrials (XLI)
    "CAT":  "XLI", "BA":   "XLI", "HON":  "XLI", "UPS":  "XLI",
    "DE":   "XLI", "GE":   "XLI",
    # Consumer Staples (XLP)
    "PG":   "XLP", "KO":   "XLP", "WMT":  "XLP", "COST": "XLP",
    "PEP":  "XLP", "PM":   "XLP",
    # Communication Services (XLC)
    "META": "XLC", "NFLX": "XLC", "DIS":  "XLC", "VZ":   "XLC",
    "T":    "XLC", "CMCSA": "XLC", "TMUS": "XLC",
}

# Market instrument symbols we fetch alongside the user's ticker.
# Keys are yfinance-compatible symbols; values are the internal column names.
MARKET_SYMBOLS: dict[str, str] = {
    "SPY":      "spy_close",
    "^VIX":     "vix_level_raw",
    "^TNX":     "tnx_level",
    "DX-Y.NYB": "dxy_level",
    "USO":      "uso_close",
    # Sector ETFs (one per sector)
    "XLK":      "xlk_close",
    "XLY":      "xly_close",
    "XLF":      "xlf_close",
    "XLV":      "xlv_close",
    "XLE":      "xle_close",
    "XLI":      "xli_close",
    "XLP":      "xlp_close",
    "XLC":      "xlc_close",
}


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index - identical to the notebook implementation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def engineer_features(
    df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Derive the 21 NextTick features from OHLCV data.

    Parameters
    ----------
    df : DataFrame
        User-ticker OHLCV, columns Open/High/Low/Close/Volume, indexed by date
        in ascending chronological order.
    market_df : DataFrame, optional
        Market context, columns as listed in ``MARKET_SYMBOLS`` values
        (spy_close, vix_level_raw, tnx_level, dxy_level, uso_close, and 8
        sector-ETF closes), indexed by date. If None, market/sector features
        are filled with zeros - useful only for a partial-feature test; models
        will produce noisy predictions in that case.
    ticker : str, optional
        Used to look up which sector ETF to use for ``sector_return`` and
        ``relative_to_sector``. Falls back to SPY when unknown.

    Returns
    -------
    DataFrame with all input columns plus the 21 engineered features. Caller
    is expected to drop NaNs from rolling/shift warmups before using rows.
    """
    out = df.copy()

    # Price / technical (6)
    out["daily_return"]  = out["Close"].pct_change()
    out["sma_10"]        = out["Close"].rolling(10).mean()
    out["sma_20"]        = out["Close"].rolling(20).mean()
    out["volatility_10"] = out["daily_return"].rolling(10).std()
    out["momentum_10"]   = out["Close"].pct_change(10)
    out["rsi_14"]        = _rsi(out["Close"], 14)

    # OHLCV-derived (5)
    out["overnight_gap"]   = (out["Open"] - out["Close"].shift(1)) / out["Close"].shift(1)
    out["intraday_return"] = (out["Close"] - out["Open"]) / out["Open"]
    out["daily_range_pct"] = (out["High"] - out["Low"]) / out["Close"]
    day_range = out["High"] - out["Low"]
    out["close_location"]  = np.where(day_range > 0,
                                       (out["Close"] - out["Low"]) / day_range,
                                       0.5)
    out["relative_volume"] = out["Volume"] / out["Volume"].rolling(20).mean()

    # Calendar (2 of the 10 market features)
    out["day_of_week"] = out.index.dayofweek  # 0=Mon ... 4=Fri
    out["month"]       = out.index.month

    # Market / macro (8 remaining) - join on date
    if market_df is not None and not market_df.empty:
        # Align and forward-fill so market features exist on every trading day
        market = market_df.reindex(out.index).ffill()

        out["spy_return"] = market["spy_close"].pct_change()
        out["vix_level"]  = market["vix_level_raw"]
        out["tnx_change"] = market["tnx_level"].diff()
        out["dxy_change"] = market["dxy_level"].pct_change()
        out["oil_return"] = market["uso_close"].pct_change()

        # Pick the correct sector ETF for this ticker.
        # Unknown tickers fall back to SPY so the feature still has a meaningful
        # value - relative_to_sector then becomes relative_to_market.
        sector_etf = TICKER_TO_SECTOR_ETF.get(
            (ticker or "").upper(), None
        )
        if sector_etf is None:
            sector_close_col = "spy_close"
        else:
            sector_close_col = f"{sector_etf.lower()}_close"

        if sector_close_col in market.columns:
            out["sector_return"] = market[sector_close_col].pct_change()
        else:
            # Final fallback: use SPY return as the sector benchmark
            out["sector_return"] = market["spy_close"].pct_change()

        # Relative features
        out["relative_to_spy"]    = out["daily_return"] - out["spy_return"]
        out["relative_to_sector"] = out["daily_return"] - out["sector_return"]
    else:
        # No market data available - set the 8 market features to 0 so the
        # columns still exist. Models will see an out-of-distribution input
        # and predictions will be unreliable; this path is a last-resort
        # safety net, not a supported mode.
        for col in ("spy_return", "vix_level", "sector_return",
                    "relative_to_spy", "relative_to_sector",
                    "tnx_change", "dxy_change", "oil_return"):
            out[col] = 0.0

    return out


def validate_input_frame(df: pd.DataFrame, min_rows: int = 30) -> None:
    """Raise ``ValueError`` if the user-ticker frame is unsuitable for inference."""
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
