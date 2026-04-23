#!/usr/bin/env python3
"""
fetch_stock_data.py — Download OHLCV data and save a CSV for the NextTick app.

Usage:
    python fetch_stock_data.py AAPL
    python fetch_stock_data.py AAPL --days 90
    python fetch_stock_data.py TSLA --output tesla.csv

Requires:
    pip install yfinance pandas-datareader pandas

Produces a CSV with columns: Date, Open, High, Low, Close, Volume
which you can drop directly onto the NextTick Flask UI.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd


# ------------------------------------------------------------------ #
# Provider: yfinance                                                  #
# ------------------------------------------------------------------ #
def fetch_yfinance(ticker: str, days: int, retries: int = 3) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        print('  [yfinance] not installed — `pip install yfinance`', file=sys.stderr)
        return None

    # Pull ~2x requested days so we have a buffer after dropping weekends/holidays.
    period_days = max(days * 2, 90)

    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(period=f'{period_days}d', auto_adjust=True)
            if df.empty:
                raise RuntimeError('empty response')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f'  [yfinance] attempt {attempt + 1} failed ({exc}); '
                      f'retrying in {wait}s...', file=sys.stderr)
                time.sleep(wait)
            else:
                print(f'  [yfinance] all {retries} attempts failed: {exc}', file=sys.stderr)
    return None


# ------------------------------------------------------------------ #
# Provider: Stooq (fallback — no rate limits)                         #
# ------------------------------------------------------------------ #
def fetch_stooq(ticker: str, days: int) -> Optional[pd.DataFrame]:
    try:
        from pandas_datareader import data as pdr
    except ImportError:
        print('  [stooq] pandas-datareader not installed — '
              '`pip install pandas-datareader`', file=sys.stderr)
        return None

    try:
        end = datetime.now()
        start = end - timedelta(days=days * 2 + 30)
        # US tickers on Stooq need a ".US" suffix
        df = pdr.DataReader(f'{ticker}.US', 'stooq', start, end).sort_index()
        if df.empty:
            raise RuntimeError('empty response')
        return df
    except Exception as exc:
        print(f'  [stooq] failed: {exc}', file=sys.stderr)
        return None


# ------------------------------------------------------------------ #
# Shaping                                                             #
# ------------------------------------------------------------------ #
REQUIRED_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']


def prepare_dataframe(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Normalise columns, drop NaNs, keep the last `days` trading days."""
    df = df.copy()
    df.columns = [str(c).title() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Provider returned data without these columns: {missing}')

    df = df[REQUIRED_COLS]
    df.index = pd.to_datetime(df.index)
    # Strip timezone info so CSV Date parses cleanly for everyone
    try:
        df.index = df.index.tz_localize(None)
    except (AttributeError, TypeError):
        pass

    df = df.dropna().sort_index().tail(days)
    df.index.name = 'Date'
    return df.reset_index()


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #
def fetch(ticker: str, days: int) -> Tuple[pd.DataFrame, str]:
    """Try yfinance first, then Stooq. Return (dataframe, source_name)."""
    print(f'  [1/2] Trying yfinance for {ticker}...')
    df = fetch_yfinance(ticker, days)
    if df is not None and not df.empty:
        return prepare_dataframe(df, days), 'yfinance'

    print(f'  [2/2] Trying Stooq for {ticker}...')
    df = fetch_stooq(ticker, days)
    if df is not None and not df.empty:
        return prepare_dataframe(df, days), 'stooq'

    raise RuntimeError(
        f'Could not fetch data for "{ticker}" from any provider.\n'
        f'Check your internet connection, try a different ticker, or wait a '
        f'few minutes if Yahoo is rate-limiting your IP.'
    )


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #
def main() -> int:
    ap = argparse.ArgumentParser(
        description='Fetch OHLCV data and save a CSV for the NextTick Flask app.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python fetch_stock_data.py AAPL\n'
               '  python fetch_stock_data.py MSFT --days 90\n'
               '  python fetch_stock_data.py TSLA --output tesla.csv',
    )
    ap.add_argument('ticker', help='Stock ticker symbol (e.g. AAPL, MSFT, TSLA)')
    ap.add_argument('--days', type=int, default=60,
                    help='Number of trading days to fetch (default: 60, minimum: 30)')
    ap.add_argument('--output', '-o', default=None,
                    help='Output CSV path (default: <ticker>_<yyyymmdd>.csv)')
    args = ap.parse_args()

    ticker = args.ticker.strip().upper()
    days = max(args.days, 30)
    output = args.output or f'{ticker}_{datetime.now().strftime("%Y%m%d")}.csv'

    print(f'Fetching {days} trading days for {ticker}...')
    try:
        df, source = fetch(ticker, days)
    except RuntimeError as exc:
        print(f'\n✗ {exc}', file=sys.stderr)
        return 1

    df.to_csv(output, index=False, date_format='%Y-%m-%d')

    print(f'\n✓ Saved {len(df)} rows from {source}')
    print(f'  Date range: {df["Date"].min().strftime("%Y-%m-%d")} '
          f'→ {df["Date"].max().strftime("%Y-%m-%d")}')
    print(f'  Last close: ${df["Close"].iloc[-1]:.2f}')
    print(f'  Output:     {output}')
    print(f'\nUpload "{output}" to the NextTick Flask app to get a forecast.')
    return 0


if __name__ == '__main__':
    sys.exit(main())