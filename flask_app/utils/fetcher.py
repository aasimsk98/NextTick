"""
Stock data fetcher for NextTick Flask app.
Pulls live OHLCV from Yahoo Finance (yfinance) with a Stooq fallback.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS    = ['Open', 'High', 'Low', 'Close', 'Volume']
TRADING_DAYS_6M  = 126   # ~6 calendar months of trading days


def _normalize(df: pd.DataFrame, days: int) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Provider returned data without columns: {missing}')
    df = df[REQUIRED_COLS]
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except (AttributeError, TypeError):
        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            pass
    df = df.dropna().sort_index().tail(days)
    df.index.name = 'Date'
    return df.reset_index()


def search_tickers(query: str, max_results: int = 8) -> list[dict]:
    """Return a list of ticker matches for a free-text query."""
    try:
        import yfinance as yf
        results = yf.Search(query, max_results=max_results, news_count=0).quotes
        out = []
        for r in results:
            sym = r.get('symbol', '')
            if not sym:
                continue
            out.append({
                'symbol':   sym,
                'name':     r.get('shortname') or r.get('longname') or sym,
                'exchange': r.get('exchDisp') or r.get('exchange', ''),
                'type':     r.get('typeDisp', ''),
            })
        return out
    except Exception as exc:
        logger.warning("Ticker search failed for %r: %s", query, exc)
        return []


def fetch_ticker_info(ticker: str) -> dict:
    """Return company metadata (name, exchange, currency, sector)."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return {
            'name':     info.get('shortName') or info.get('longName') or ticker,
            'exchange': info.get('exchange', ''),
            'currency': info.get('currency', 'USD'),
            'sector':   info.get('sector', ''),
        }
    except Exception:
        return {'name': ticker, 'exchange': '', 'currency': 'USD', 'sector': ''}


def fetch_ohlcv(
    ticker: str,
    days: int = TRADING_DAYS_6M,
    retries: int = 3,
) -> Tuple[pd.DataFrame, str]:
    """
    Fetch `days` trading days of OHLCV for `ticker`.
    Returns (DataFrame, source_name).
    DataFrame columns: Date, Open, High, Low, Close, Volume — sorted ascending.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    ticker      = ticker.strip().upper()
    period_days = max(days * 2, 365)
    last_exc: Optional[Exception] = None

    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(period=f'{period_days}d', auto_adjust=True)
            if df.empty:
                raise RuntimeError(f'No data returned for "{ticker}"')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return _normalize(df, days), 'Yahoo Finance'
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    # Stooq fallback
    try:
        from pandas_datareader import data as pdr
        end   = datetime.now()
        start = end - timedelta(days=period_days)
        df    = pdr.DataReader(f'{ticker}.US', 'stooq', start, end).sort_index()
        if not df.empty:
            return _normalize(df, days), 'Stooq'
    except Exception:
        pass

    raise RuntimeError(
        f'Could not fetch data for "{ticker}". '
        f'Verify the ticker symbol and check your internet connection. '
        f'(Last error: {last_exc})'
    )
