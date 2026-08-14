"""Entry-time features and MAE/MFE. All features use bars strictly before entry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MARKET_TICKER, PROC_DIR, RTH_END, RTH_START, SECTOR_TICKER, TZ
from .market_data import load_bars


def _truthy(v) -> bool:
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return bool(v)


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    au = up.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    ad = down.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = au / ad.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(span=period, adjust=False, min_periods=period).mean()


def session_date(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.tz_convert(TZ).date, index=idx)


def add_indicators(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    c = out["close"]
    out["ema9"] = ema(c, 9)
    out["ema20"] = ema(c, 20)
    out["ema50"] = ema(c, 50)
    out["ema200"] = ema(c, 200)
    out["rsi5"] = rsi(c, 5)
    out["rsi14"] = rsi(c, 14)
    out["atr14"] = atr(out, 14)
    out["ret_1"] = c.pct_change()
    out["vol_roll"] = out["ret_1"].rolling(20, min_periods=10).std()
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    dates = session_date(out.index)
    out["session"] = dates.values
    out["tpv"] = typical * out["volume"]
    grp = out.groupby("session", sort=False)
    sess_vol = grp["volume"].cumsum().replace(0, np.nan)
    out["session_vwap"] = grp["tpv"].cumsum() / sess_vol
    out["session_open"] = grp["open"].transform("first")
    out["session_high"] = grp["high"].cummax()
    out["session_low"] = grp["low"].cummin()
    out["session_ret"] = c / out["session_open"] - 1.0
    vol_mean = grp["volume"].transform(lambda s: s.shift(1).expanding().mean())
    out["rvol"] = out["volume"] / vol_mean.replace(0, np.nan)
    out["rvol20"] = out["volume"] / out["volume"].rolling(20, min_periods=5).median().replace(0, np.nan)
    # previous session high/low via daily group last
    sess_hl = grp.agg(s_high=("high", "max"), s_low=("low", "min"), s_close=("close", "last"))
    sess_hl["pdh"] = sess_hl["s_high"].shift(1)
    sess_hl["pdl"] = sess_hl["s_low"].shift(1)
    sess_hl["pdc"] = sess_hl["s_close"].shift(1)
    out = out.join(sess_hl[["pdh", "pdl", "pdc"]], on="session")
    out["gap_pct"] = np.where(
        out["pdc"].notna(),
        out["session_open"] / out["pdc"] - 1.0,
        np.nan,
    )
    minutes = out.index.hour * 60 + out.index.minute
    rth0 = RTH_START[0] * 60 + RTH_START[1]
    or_end = rth0 + 30
    rth1 = RTH_END[0] * 60 + RTH_END[1]
    out["is_rth"] = (minutes >= rth0) & (minutes < rth1)
    out["in_or"] = (minutes >= rth0) & (minutes < or_end)
    or_hl = (
        out.loc[out["in_or"]]
        .groupby("session")
        .agg(or_high=("high", "max"), or_low=("low", "min"))
    )
    out = out.join(or_hl, on="session")
    # OR high/low are not knowable until the opening range completes.
    out.loc[out["in_or"] | ~out["is_rth"], ["or_high", "or_low"]] = np.nan
    out["dist_ema20"] = c / out["ema20"] - 1.0
    out["dist_vwap"] = c / out["session_vwap"] - 1.0
    out["dist_sess_high"] = c / out["session_high"] - 1.0
    out["dist_sess_low"] = c / out["session_low"] - 1.0
    out["hh"] = out["high"] > out["high"].shift(1)
    out["hl"] = out["low"] > out["low"].shift(1)
    out["lh"] = out["high"] < out["high"].shift(1)
    out["ll"] = out["low"] < out["low"].shift(1)
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["large_candle"] = rng > rng.rolling(20, min_periods=10).median() * 1.8
    out["dir_up"] = out["close"] > out["open"]
    out["dir_dn"] = out["close"] < out["open"]
    out["consec_up"] = out["dir_up"].groupby((~out["dir_up"]).cumsum()).cumcount() + 1
    out["consec_up"] = np.where(out["dir_up"], out["consec_up"], 0)
    out["consec_dn"] = out["dir_dn"].groupby((~out["dir_dn"]).cumsum()).cumcount() + 1
    out["consec_dn"] = np.where(out["dir_dn"], out["consec_dn"], 0)
    atrp = out["atr14"] / c
    out["vol_expand"] = atrp > atrp.rolling(50, min_periods=20).median() * 1.25
    out["vol_contract"] = atrp < atrp.rolling(50, min_periods=20).median() * 0.8
    out["ema_bull"] = (out["ema9"] > out["ema20"]) & (out["ema20"] > out["ema50"])
    out["ema_bear"] = (out["ema9"] < out["ema20"]) & (out["ema20"] < out["ema50"])
    out["above_vwap"] = c > out["session_vwap"]
    out["vwap_reclaim"] = out["above_vwap"] & ~out["above_vwap"].shift(1).fillna(False)
    out["vwap_reject"] = (~out["above_vwap"]) & out["above_vwap"].shift(1).fillna(False)
    out["above_ema20"] = c > out["ema20"]
    out["ema20_reclaim"] = out["above_ema20"] & ~out["above_ema20"].shift(1).fillna(False)
    out["ema20_reject"] = (~out["above_ema20"]) & out["above_ema20"].shift(1).fillna(False)
    out["breakout"] = out["high"] >= out["high"].rolling(20, min_periods=10).max().shift(1)
    out["breakdown"] = out["low"] <= out["low"].rolling(20, min_periods=10).min().shift(1)
    out["orb"] = (out["or_high"].notna()) & (c > out["or_high"]) & (minutes >= or_end)
    out["or_fail"] = (out["or_high"].notna()) & (
        ((c < out["or_low"]) & (out["high"].rolling(3).max() >= out["or_high"]))
        | ((c > out["or_high"]) & (out["low"].rolling(3).min() <= out["or_low"]))
    )
    # breakout failure: 20-bar high taken then close back inside
    roll_high = out["high"].rolling(20, min_periods=10).max().shift(1)
    out["breakout_fail"] = (out["high"] >= roll_high) & (c < roll_high)
    out["interval"] = interval
    # Yahoo labels bars at interval start. Shift to close so features are knowable.
    shift = {"1m": "1min", "5m": "5min", "15m": "15min", "60m": "60min"}
    if interval in shift:
        out.index = out.index + pd.Timedelta(shift[interval])
    elif interval == "1d":
        idx = out.index.tz_convert(TZ)
        out.index = idx.normalize() + pd.Timedelta(hours=16)
    return out


_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def bars_with_features(ticker: str, interval: str) -> pd.DataFrame:
    key = (ticker, interval)
    if key not in _CACHE:
        raw = load_bars(ticker, interval)
        _CACHE[key] = add_indicators(raw, interval) if not raw.empty else raw
    return _CACHE[key]


def last_bar_before(df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    if df.empty or pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(TZ)
    else:
        ts = ts.tz_convert(TZ)
    idx = df.index.searchsorted(ts, side="left")
    if idx <= 0:
        return None
    return df.iloc[idx - 1]


def return_lookback(df: pd.DataFrame, ts: pd.Timestamp, minutes: int) -> float:
    if df.empty:
        return np.nan
    bar = last_bar_before(df, ts)
    if bar is None:
        return np.nan
    start = ts - pd.Timedelta(minutes=minutes)
    prior = last_bar_before(df, start)
    if prior is None or prior["close"] == 0:
        return np.nan
    return float(bar["close"] / prior["close"] - 1.0)


FEATURE_COLS = [
    "feat_interval",
    "und_close",
    "und_ema9",
    "und_ema20",
    "und_ema50",
    "und_ema200",
    "und_rsi5",
    "und_rsi14",
    "und_vwap",
    "und_atr14",
    "und_rvol",
    "und_vol_roll",
    "und_session_ret",
    "und_gap_pct",
    "und_dist_ema20",
    "und_dist_vwap",
    "und_dist_sess_high",
    "und_dist_sess_low",
    "und_pdh",
    "und_pdl",
    "und_or_high",
    "und_or_low",
    "und_ema_bull",
    "und_ema_bear",
    "und_above_vwap",
    "und_vwap_reclaim",
    "und_vwap_reject",
    "und_ema20_reclaim",
    "und_ema20_reject",
    "und_hh",
    "und_hl",
    "und_lh",
    "und_ll",
    "und_breakout",
    "und_breakdown",
    "und_breakout_fail",
    "und_orb",
    "und_or_fail",
    "und_large_candle",
    "und_consec_up",
    "und_consec_dn",
    "und_vol_expand",
    "und_vol_contract",
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "ret_1h",
    "ret_1d",
    "spy_session_ret",
    "soxx_session_ret",
    "sector_confirm",
    "extended_2atr",
    "trend_vs_chop",
]


def pick_interval_for_time(ticker: str, ts: pd.Timestamp) -> tuple[str, pd.DataFrame]:
    for interval in ("5m", "15m", "1m", "60m", "1d"):
        df = bars_with_features(ticker, interval)
        if df.empty:
            continue
        bar = last_bar_before(df, ts)
        if bar is None:
            continue
        # Require the last bar to be reasonably close to entry (no multi-day stale 5m).
        lag_min = (ts - bar.name).total_seconds() / 60.0
        max_lag = {"1m": 15, "5m": 45, "15m": 90, "60m": 400, "1d": 60 * 24 * 3}[interval]
        if lag_min <= max_lag:
            return interval, df
    return "none", pd.DataFrame()


def extract_underlying_features(underlying: str, ts: pd.Timestamp) -> dict:
    interval, df = pick_interval_for_time(underlying, ts)
    feat = {c: np.nan for c in FEATURE_COLS}
    feat["feat_interval"] = interval
    if df.empty:
        return feat
    bar = last_bar_before(df, ts)
    if bar is None:
        return feat
    feat.update(
        {
            "und_close": bar.get("close", np.nan),
            "und_ema9": bar.get("ema9", np.nan),
            "und_ema20": bar.get("ema20", np.nan),
            "und_ema50": bar.get("ema50", np.nan),
            "und_ema200": bar.get("ema200", np.nan),
            "und_rsi5": bar.get("rsi5", np.nan),
            "und_rsi14": bar.get("rsi14", np.nan),
            "und_vwap": bar.get("session_vwap", np.nan),
            "und_atr14": bar.get("atr14", np.nan),
            "und_rvol": bar.get("rvol", np.nan),
            "und_vol_roll": bar.get("vol_roll", np.nan),
            "und_session_ret": bar.get("session_ret", np.nan),
            "und_gap_pct": bar.get("gap_pct", np.nan),
            "und_dist_ema20": bar.get("dist_ema20", np.nan),
            "und_dist_vwap": bar.get("dist_vwap", np.nan),
            "und_dist_sess_high": bar.get("dist_sess_high", np.nan),
            "und_dist_sess_low": bar.get("dist_sess_low", np.nan),
            "und_pdh": bar.get("pdh", np.nan),
            "und_pdl": bar.get("pdl", np.nan),
            "und_or_high": bar.get("or_high", np.nan),
            "und_or_low": bar.get("or_low", np.nan),
            "und_ema_bull": _truthy(bar.get("ema_bull", False)),
            "und_ema_bear": _truthy(bar.get("ema_bear", False)),
            "und_above_vwap": _truthy(bar.get("above_vwap", False)),
            "und_vwap_reclaim": _truthy(bar.get("vwap_reclaim", False)),
            "und_vwap_reject": _truthy(bar.get("vwap_reject", False)),
            "und_ema20_reclaim": _truthy(bar.get("ema20_reclaim", False)),
            "und_ema20_reject": _truthy(bar.get("ema20_reject", False)),
            "und_hh": _truthy(bar.get("hh", False)),
            "und_hl": _truthy(bar.get("hl", False)),
            "und_lh": _truthy(bar.get("lh", False)),
            "und_ll": _truthy(bar.get("ll", False)),
            "und_breakout": _truthy(bar.get("breakout", False)),
            "und_breakdown": _truthy(bar.get("breakdown", False)),
            "und_breakout_fail": _truthy(bar.get("breakout_fail", False)),
            "und_orb": _truthy(bar.get("orb", False)),
            "und_or_fail": _truthy(bar.get("or_fail", False)),
            "und_large_candle": _truthy(bar.get("large_candle", False)),
            "und_consec_up": bar.get("consec_up", np.nan),
            "und_consec_dn": bar.get("consec_dn", np.nan),
            "und_vol_expand": _truthy(bar.get("vol_expand", False)),
            "und_vol_contract": _truthy(bar.get("vol_contract", False)),
        }
    )
    feat["ret_5m"] = return_lookback(df, ts, 5)
    feat["ret_15m"] = return_lookback(df, ts, 15)
    feat["ret_30m"] = return_lookback(df, ts, 30)
    feat["ret_1h"] = return_lookback(df, ts, 60)
    daily = bars_with_features(underlying, "1d")
    feat["ret_1d"] = return_lookback(daily if not daily.empty else df, ts, 60 * 24)
    spy = bars_with_features(MARKET_TICKER, interval if interval != "none" else "1d")
    soxx = bars_with_features(SECTOR_TICKER, interval if interval != "none" else "1d")
    spy_bar = last_bar_before(spy, ts) if not spy.empty else None
    soxx_bar = last_bar_before(soxx, ts) if not soxx.empty else None
    feat["spy_session_ret"] = spy_bar["session_ret"] if spy_bar is not None else np.nan
    feat["soxx_session_ret"] = soxx_bar["session_ret"] if soxx_bar is not None else np.nan
    und_ret = feat["und_session_ret"]
    sec_ret = feat["soxx_session_ret"]
    if pd.notna(und_ret) and pd.notna(sec_ret):
        feat["sector_confirm"] = float(np.sign(und_ret) == np.sign(sec_ret) and und_ret != 0)
    atr14 = feat["und_atr14"]
    close = feat["und_close"]
    vwap = feat["und_vwap"]
    if pd.notna(atr14) and pd.notna(close) and pd.notna(vwap) and atr14 > 0:
        feat["extended_2atr"] = float(abs(close - vwap) > 2 * atr14)
    if feat["und_ema_bull"] or feat["und_ema_bear"]:
        feat["trend_vs_chop"] = 1.0 if (feat["und_ema_bull"] or feat["und_ema_bear"]) else 0.0
        if feat["und_vol_contract"] and not (feat["und_hh"] or feat["und_ll"]):
            feat["trend_vs_chop"] = 0.0
    else:
        feat["trend_vs_chop"] = 0.0
    return feat


_MAX_ENTRY_GAP = {
    "1m": pd.Timedelta(minutes=5),
    "5m": pd.Timedelta(minutes=20),
    "15m": pd.Timedelta(minutes=45),
    "60m": pd.Timedelta(hours=3),
    "1d": pd.Timedelta(days=2),
}

_BAR_LEN = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "60m": pd.Timedelta(minutes=60),
    "1d": pd.Timedelta(days=1),
}


def _scale_ohlc_to_fill(df: pd.DataFrame, ts: pd.Timestamp, fill_px: float) -> tuple[pd.DataFrame, float]:
    """Yahoo OHLC is often split-adjusted; fills are not. Rescale to the fill."""
    if df.empty or not fill_px:
        return df, 1.0
    bar = last_bar_before(df, ts + pd.Timedelta(seconds=1))
    if bar is None or not np.isfinite(bar.get("close", np.nan)) or bar["close"] == 0:
        return df, 1.0
    scale = float(fill_px) / float(bar["close"])
    if abs(scale - 1.0) < 0.15:
        return df, 1.0
    if scale < 0.02 or scale > 80:
        return df, 1.0
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col] * scale
    return out, scale


def excursion(symbol: str, entry: pd.Timestamp, exit: pd.Timestamp, entry_px: float) -> dict:
    out = {
        "mae_pct": np.nan,
        "mfe_pct": np.nan,
        "max_unrealized_pnl_pct": np.nan,
        "max_unrealized_loss_pct": np.nan,
        "minutes_to_mfe": np.nan,
        "minutes_to_mae": np.nan,
        "path_interval": "none",
        "path_bars": 0,
        "path_price_scale": 1.0,
    }
    if pd.isna(entry) or pd.isna(exit) or not entry_px:
        return out
    for interval in ("1m", "5m", "15m", "60m", "1d"):
        df = load_bars(symbol, interval)
        if df.empty:
            continue
        max_gap = _MAX_ENTRY_GAP[interval]
        bar_len = _BAR_LEN[interval]
        # Yahoo labels bars at interval start. Keep bars that overlap the hold,
        # not bars that closed entirely before the fill.
        overlap = (df.index + bar_len > entry) & (df.index <= exit)
        window = df.loc[overlap]
        if window.empty:
            continue
        if window.index[0] - entry > max_gap:
            continue
        scaled_df, scale = _scale_ohlc_to_fill(df, entry, entry_px)
        scaled = scaled_df.loc[window.index]
        highs = scaled["high"].astype(float)
        lows = scaled["low"].astype(float)
        mfe = (highs.max() / entry_px - 1.0) * 100.0
        mae = (lows.min() / entry_px - 1.0) * 100.0
        mfe_t = highs.idxmax()
        mae_t = lows.idxmin()
        out.update(
            {
                "mae_pct": mae,
                "mfe_pct": mfe,
                "max_unrealized_pnl_pct": mfe,
                "max_unrealized_loss_pct": mae,
                "minutes_to_mfe": max(0.0, (mfe_t - entry).total_seconds() / 60.0),
                "minutes_to_mae": max(0.0, (mae_t - entry).total_seconds() / 60.0),
                "path_interval": interval,
                "path_bars": int(len(window)),
                "path_price_scale": scale,
            }
        )
        return out
    return out


def annotate_trades(trips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, tr in trips.iterrows():
        feat = extract_underlying_features(tr["underlying"], tr["entry_time"])
        # SK Hynix: also compute SOXX proxy if Korean bars are stale/unavailable
        if tr.get("family") == "SKHYNIX" and feat["feat_interval"] in {"none", "1d"}:
            proxy = extract_underlying_features("SOXX", tr["entry_time"])
            feat = {**feat, **{k: v for k, v in proxy.items() if k != "feat_interval"}}
            feat["feat_interval"] = f"soxx_proxy:{proxy['feat_interval']}"
        path = excursion(tr["symbol"], tr["entry_time"], tr["exit_time"], tr["entry_price"])
        rows.append({**tr.to_dict(), **feat, **path})
    out = pd.DataFrame(rows)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC_DIR / "trades_featured.parquet", index=False)
    out.to_csv(PROC_DIR / "trades_featured.csv", index=False)
    return out
