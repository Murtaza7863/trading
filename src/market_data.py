"""Download and cache OHLCV. Never fabricate missing bars."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import (
    ETF_TICKERS,
    MARKET_DIR,
    MARKET_TICKER,
    SECTOR_TICKER,
    UNDERLYING_TICKERS,
    YF_DAILY_START,
    YF_END,
    YF_HOURLY_START,
    TZ,
)

INTERVALS = {
    "1m": {"period": "7d", "interval": "1m", "prepost": True},
    "5m": {"period": "60d", "interval": "5m", "prepost": True},
    "15m": {"period": "60d", "interval": "15m", "prepost": True},
    "60m": {
        "start": YF_HOURLY_START,
        "end": YF_END,
        "interval": "60m",
        "prepost": True,
    },
    "1d": {"start": YF_DAILY_START, "end": YF_END, "interval": "1d", "prepost": False},
}


def all_tickers() -> list[str]:
    tickers = set(ETF_TICKERS) | set(UNDERLYING_TICKERS) | {SECTOR_TICKER, MARKET_TICKER}
    return sorted(tickers)


def _history(ticker: str, spec: dict) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    kwargs = dict(
        interval=spec["interval"],
        auto_adjust=False,
        actions=True,
        timeout=60,
    )
    if "period" in spec:
        kwargs["period"] = spec["period"]
    else:
        kwargs["start"] = spec["start"]
        kwargs["end"] = spec["end"]
    if spec.get("prepost"):
        kwargs["prepost"] = True
    df = t.history(**kwargs)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "splits",
        }
    )
    keep = [c for c in ["open", "high", "low", "close", "volume", "dividends", "splits"] if c in df.columns]
    df = df[keep].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert(TZ)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"], how="any")
    return df


def cache_path(ticker: str, interval: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("/", "_")
    return MARKET_DIR / interval / f"{safe}.parquet"


def download_all(force: bool = False, sleep_s: float = 0.35) -> dict:
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"ok": {}, "empty": {}, "error": {}}
    for interval, spec in INTERVALS.items():
        (MARKET_DIR / interval).mkdir(parents=True, exist_ok=True)
        for ticker in all_tickers():
            path = cache_path(ticker, interval)
            key = f"{ticker}:{interval}"
            if path.exists() and not force:
                try:
                    df = pd.read_parquet(path)
                    if df.empty:
                        manifest["empty"][key] = "cached_empty"
                    else:
                        manifest["ok"][key] = {
                            "rows": int(len(df)),
                            "start": str(df.index.min()),
                            "end": str(df.index.max()),
                        }
                    continue
                except Exception as exc:
                    manifest["error"][key] = f"cache_read: {exc}"
            try:
                df = _history(ticker, spec)
                df.to_parquet(path)
                if df.empty:
                    manifest["empty"][key] = "yahoo_empty"
                else:
                    manifest["ok"][key] = {
                        "rows": int(len(df)),
                        "start": str(df.index.min()),
                        "end": str(df.index.max()),
                    }
            except Exception as exc:
                manifest["error"][key] = str(exc)
            time.sleep(sleep_s)
    (MARKET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_bars(ticker: str, interval: str) -> pd.DataFrame:
    path = cache_path(ticker, interval)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is None:
        df.index = idx.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")
    else:
        df.index = idx.tz_convert(TZ)
    return df.sort_index()


def best_intraday(ticker: str) -> tuple[str, pd.DataFrame]:
    for interval in ("1m", "5m", "15m", "60m", "1d"):
        df = load_bars(ticker, interval)
        if not df.empty:
            return interval, df
    return "none", pd.DataFrame()
