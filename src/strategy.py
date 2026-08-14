"""Regime / setup / risk engine and baseline backtests. Research only — not live."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import (
    ASSUMED_EQUITY,
    COOLDOWN_MINUTES,
    FIG_DIR,
    FLAT_MINUTES_BEFORE_CLOSE,
    MAX_HOLD_MINUTES_V1,
    MAX_PORTFOLIO_RISK_PCT,
    MAX_SIMULTANEOUS,
    PROC_DIR,
    RANDOM_SEED,
    RISK_PCT_PER_TRADE,
    RTH_END,
    RTH_START,
    SLIPPAGE_BPS_DEFAULT,
    SLIPPAGE_BPS_GRID,
    TABLE_DIR,
    TZ,
    WALK_FORWARD_OOS_FRAC,
)
from .features import add_indicators, last_bar_before
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

UNIVERSE = [
    ("MU", 2.0),
    ("NVDA", 2.0),
    ("SOXX", 3.0),
]


def _as_tz(x):
    if x is None:
        return None
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        return t.tz_localize(TZ)
    return t.tz_convert(TZ)


@dataclass
class Params:
    stop_atr: float = 1.5
    max_hold_min: int = MAX_HOLD_MINUTES_V1
    cooldown_min: int = COOLDOWN_MINUTES
    rvol_min: float = 0.8
    max_ext_atr: float = 1.5
    flatten_before_close_min: int = FLAT_MINUTES_BEFORE_CLOSE
    require_or_complete: bool = True
    no_overnight: bool = True
    slippage_bps: float = SLIPPAGE_BPS_DEFAULT
    risk_pct: float = RISK_PCT_PER_TRADE
    max_positions: int = MAX_SIMULTANEOUS
    style: str = "trend"  # trend | fade
    use_stops: bool = True
    tickers: tuple[str, ...] | None = None


def _size_position(close: float, atr14: float, lev: float, p: Params):
    """Size so a stop_atr move on the underlying loses `risk_pct` after leverage."""
    stop_dist = p.stop_atr * float(atr14)
    risk = ASSUMED_EQUITY * p.risk_pct
    shares = risk / (stop_dist * abs(float(lev)))
    notional = shares * float(close)
    return shares, stop_dist, risk, notional


def _rth_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """True for Yahoo start-labels whose bar opens during RTH."""
    minutes = idx.hour * 60 + idx.minute
    a = RTH_START[0] * 60 + RTH_START[1]
    b = RTH_END[0] * 60 + RTH_END[1]
    return (minutes >= a) & (minutes < b)


def _close_rth_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """True for bar-close timestamps that do not extend past the cash close."""
    minutes = idx.hour * 60 + idx.minute
    a = RTH_START[0] * 60 + RTH_START[1]
    b = RTH_END[0] * 60 + RTH_END[1]
    return (minutes > a) & (minutes <= b)


def _last_session_bar(df: pd.DataFrame, ts: pd.Timestamp) -> bool:
    i = df.index.searchsorted(ts, side="right")
    if i >= len(df):
        return True
    return df.index[i].date() != ts.date()


def _force_eod_px(df: pd.DataFrame, ts: pd.Timestamp, pos: dict) -> tuple[float, pd.Timestamp] | None:
    """If this name has no bar at `ts` but the session is over, flatten on last close."""
    bar = last_bar_before(df, ts)
    if bar is None:
        return None
    return float(bar["close"]), bar.name


_FRAME_CACHE: dict[str, tuple[str, pd.DataFrame]] = {}


def load_underlying_frame(ticker: str) -> tuple[str, pd.DataFrame]:
    if ticker in _FRAME_CACHE:
        return _FRAME_CACHE[ticker]
    five_raw = load_bars(ticker, "5m")
    hour_raw = load_bars(ticker, "60m")
    five = pd.DataFrame()
    hour = pd.DataFrame()
    # Mask RTH on Yahoo start-labels *before* add_indicators shifts to bar close,
    # otherwise the last premarket bar is shifted into the first RTH timestamp.
    if not hour_raw.empty:
        hour_raw = hour_raw[_rth_mask(hour_raw.index)].copy()
        if not hour_raw.empty:
            hour = add_indicators(hour_raw, "60m")
            hour = hour[_close_rth_mask(hour.index)].copy()
    if not five_raw.empty:
        five_raw = five_raw[_rth_mask(five_raw.index)].copy()
        if not five_raw.empty:
            five = add_indicators(five_raw, "5m")
            five = five[_close_rth_mask(five.index)].copy()
    if not five.empty and not hour.empty:
        hour = hour[hour.index < five.index.min()]
        stitched = pd.concat([hour, five]).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="last")]
        result = ("60m+5m", stitched)
    elif not five.empty:
        result = ("5m", five)
    elif not hour.empty:
        result = ("60m", hour)
    else:
        raw = load_bars(ticker, "1d")
        if raw.empty:
            result = ("none", pd.DataFrame())
        else:
            result = ("1d", add_indicators(raw, "1d"))
    _FRAME_CACHE[ticker] = result
    return result


def regime_of(bar: pd.Series) -> str:
    ema_bull = _truthy(bar.get("ema_bull", False))
    ema_bear = _truthy(bar.get("ema_bear", False))
    above = _truthy(bar.get("above_vwap", False))
    expand = _truthy(bar.get("vol_expand", False))
    contract = _truthy(bar.get("vol_contract", False))
    hh = _truthy(bar.get("hh", False))
    hl = _truthy(bar.get("hl", False))
    lh = _truthy(bar.get("lh", False))
    ll = _truthy(bar.get("ll", False))
    if ema_bull and above and (hh or hl) and not contract:
        return "bullish_trend"
    if ema_bear and (not above) and (lh or ll) and not contract:
        return "bearish_trend"
    if contract and not expand:
        return "choppy_no_trade"
    if ema_bull and above:
        return "bullish_trend"
    if ema_bear and not above:
        return "bearish_trend"
    return "choppy_no_trade"


def long_setup(bar: pd.Series, p: Params) -> bool:
    close = bar.get("close", np.nan)
    vwap = bar.get("session_vwap", np.nan)
    atr14 = bar.get("atr14", np.nan)
    ema20 = bar.get("ema20", np.nan)
    if not np.isfinite(close) or not np.isfinite(vwap) or not np.isfinite(atr14):
        return False
    if close <= vwap:
        return False
    if not _truthy(bar.get("ema_bull", False)) and not (
        bar.get("ema9", 0) > bar.get("ema20", 0)
    ):
        return False
    ext = abs(close - vwap) / atr14 if atr14 else 99
    if ext > p.max_ext_atr:
        return False
    rvol = bar.get("rvol20", bar.get("rvol", np.nan))
    if np.isfinite(rvol) and rvol < p.rvol_min:
        return False
    if _truthy(bar.get("or_fail", False)) or _truthy(bar.get("breakout_fail", False)):
        return False
    mom = _truthy(bar.get("hh", False)) or _truthy(bar.get("dir_up", False))
    if not mom and not _truthy(bar.get("vwap_reclaim", False)) and not _truthy(bar.get("ema20_reclaim", False)):
        return False
    if np.isfinite(ema20) and close < ema20 and not _truthy(bar.get("ema20_reclaim", False)):
        return False
    return True


def short_setup(bar: pd.Series, p: Params) -> bool:
    close = bar.get("close", np.nan)
    vwap = bar.get("session_vwap", np.nan)
    atr14 = bar.get("atr14", np.nan)
    ema20 = bar.get("ema20", np.nan)
    if not np.isfinite(close) or not np.isfinite(vwap) or not np.isfinite(atr14):
        return False
    if close >= vwap:
        return False
    if not _truthy(bar.get("ema_bear", False)) and not (
        bar.get("ema9", 1) < bar.get("ema20", 0)
    ):
        return False
    ext = abs(close - vwap) / atr14 if atr14 else 99
    if ext > p.max_ext_atr:
        return False
    rvol = bar.get("rvol20", bar.get("rvol", np.nan))
    if np.isfinite(rvol) and rvol < p.rvol_min:
        return False
    if _truthy(bar.get("or_fail", False)) or _truthy(bar.get("breakout_fail", False)):
        return False
    mom = _truthy(bar.get("ll", False)) or _truthy(bar.get("dir_dn", False))
    if not mom and not _truthy(bar.get("vwap_reject", False)) and not _truthy(bar.get("ema20_reject", False)):
        return False
    if np.isfinite(ema20) and close > ema20 and not _truthy(bar.get("ema20_reject", False)):
        return False
    return True


def fade_long(bar: pd.Series, p: Params) -> bool:
    close = bar.get("close", np.nan)
    vwap = bar.get("session_vwap", np.nan)
    atr14 = bar.get("atr14", np.nan)
    if not np.isfinite(close) or not np.isfinite(vwap) or not np.isfinite(atr14) or atr14 <= 0:
        return False
    if close >= vwap:
        return False
    ext = (vwap - close) / atr14
    return 0.3 <= ext <= p.max_ext_atr


def fade_short(bar: pd.Series, p: Params) -> bool:
    close = bar.get("close", np.nan)
    vwap = bar.get("session_vwap", np.nan)
    atr14 = bar.get("atr14", np.nan)
    if not np.isfinite(close) or not np.isfinite(vwap) or not np.isfinite(atr14) or atr14 <= 0:
        return False
    if close <= vwap:
        return False
    ext = (close - vwap) / atr14
    return 0.3 <= ext <= p.max_ext_atr


def pick_side(bar: pd.Series, p: Params) -> str | None:
    if p.style == "fade":
        if fade_long(bar, p):
            return "long"
        if fade_short(bar, p):
            return "short"
        return None
    rg = regime_of(bar)
    if rg == "bullish_trend" and long_setup(bar, p):
        return "long"
    if rg == "bearish_trend" and short_setup(bar, p):
        return "short"
    return None


def _universe(p: Params) -> list[tuple[str, float]]:
    if not p.tickers:
        return list(UNIVERSE)
    levmap = dict(UNIVERSE)
    return [(t, levmap[t]) for t in p.tickers if t in levmap]


def _cost(notional: float, bps: float) -> float:
    return abs(notional) * bps / 10_000.0


def _metrics_from_trades(tr: pd.DataFrame, equity: pd.Series) -> dict:
    if tr.empty:
        return {
            "n": 0,
            "total_pnl": 0.0,
            "expectancy": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "avg_win_avg_loss": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "max_drawdown_pct": np.nan,
        }
    pnl = tr["pnl"].to_numpy(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gw = wins.sum() if len(wins) else 0.0
    gl = -losses.sum() if len(losses) else 0.0
    eq = equity.dropna()
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = np.nan
    sortino = np.nan
    if len(rets) > 5 and rets.std() > 0:
        sharpe = float(np.sqrt(252) * rets.mean() / rets.std())
        downside = rets[rets < 0]
        if len(downside) and downside.std() > 0:
            sortino = float(np.sqrt(252) * rets.mean() / downside.std())
    peak = eq.cummax()
    dd = eq - peak
    dd_pct = eq / peak - 1.0
    return {
        "n": int(len(tr)),
        "total_pnl": float(pnl.sum()),
        "expectancy": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(gw / gl) if gl > 0 else np.nan,
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "avg_win_avg_loss": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd.min()) if len(dd) else np.nan,
        "max_drawdown_pct": float(dd_pct.min()) if len(dd_pct) else np.nan,
        "ending_equity": float(eq.iloc[-1]) if len(eq) else ASSUMED_EQUITY,
    }


def simulate(p: Params, start=None, end=None) -> tuple[pd.DataFrame, pd.Series, dict]:
    frames = {}
    intervals = {}
    for ticker, lev in _universe(p):
        interval, df = load_underlying_frame(ticker)
        if df.empty:
            continue
        if start is not None:
            df = df[df.index >= _as_tz(start)]
        if end is not None:
            df = df[df.index < _as_tz(end)]
        frames[ticker] = (lev, df)
        intervals[ticker] = interval
    if not frames:
        return pd.DataFrame(), pd.Series(dtype=float), {"error": "no_bars"}

    index = None
    for _, df in frames.values():
        index = df.index if index is None else index.union(df.index)
    index = index.sort_values()

    cash = ASSUMED_EQUITY
    positions = {}  # ticker -> dict
    cooldown_until = {t: None for t in frames}
    trades = []
    equity_marks = []

    rth_end_min = RTH_END[0] * 60 + RTH_END[1]
    or_end_min = RTH_START[0] * 60 + RTH_START[1] + 30
    flatten_min = rth_end_min - p.flatten_before_close_min

    for ts in index:
        minute = ts.hour * 60 + ts.minute
        # mark / manage
        to_close = []
        for ticker, pos in list(positions.items()):
            lev, df = frames[ticker]
            if ts not in df.index:
                if p.no_overnight and (
                    minute >= flatten_min or ts.date() != pd.Timestamp(pos["entry_time"]).date()
                ):
                    forced = _force_eod_px(df, ts, pos)
                    if forced is not None:
                        px, ets = forced
                        to_close.append((ticker, px, "eod", ets, px))
                continue
            bar = df.loc[ts]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
            reason = None
            exit_px = close
            if p.use_stops:
                if pos["side"] == "long":
                    if np.isfinite(pos["stop"]) and low <= pos["stop"]:
                        reason = "stop"
                        exit_px = pos["stop"]
                    elif np.isfinite(pos["target"]) and high >= pos["target"]:
                        reason = "target"
                        exit_px = pos["target"]
                else:
                    if np.isfinite(pos["stop"]) and high >= pos["stop"]:
                        reason = "stop"
                        exit_px = pos["stop"]
                    elif np.isfinite(pos["target"]) and low <= pos["target"]:
                        reason = "target"
                        exit_px = pos["target"]
            held = (ts - pos["entry_time"]).total_seconds() / 60.0
            if reason is None and held >= p.max_hold_min:
                reason = "time"
                exit_px = close
            if reason is None and p.no_overnight and (
                minute >= flatten_min or _last_session_bar(df, ts)
            ):
                reason = "eod"
                exit_px = close
            if reason:
                to_close.append((ticker, exit_px, reason, ts, close))

        for ticker, exit_px, reason, ets, last in to_close:
            pos = positions.pop(ticker)
            lev = frames[ticker][0]
            slip = _cost(pos["notional"], p.slippage_bps)
            move = (exit_px - pos["entry_px"]) * pos["shares"] * pos["sign"] * lev
            pnl = move - slip - pos["entry_cost"]
            cash += pnl
            trades.append(
                {
                    "ticker": ticker,
                    "leverage": lev,
                    "side": pos["side"],
                    "entry_time": pos["entry_time"],
                    "exit_time": ets,
                    "entry_px": pos["entry_px"],
                    "exit_px": exit_px,
                    "shares": pos["shares"],
                    "notional": pos["notional"],
                    "pnl": pnl,
                    "return_pct": 100.0 * pnl / pos["notional"] if pos["notional"] else np.nan,
                    "reason": reason,
                    "regime": pos["regime"],
                    "hold_minutes": (ets - pos["entry_time"]).total_seconds() / 60.0,
                    "stop": pos["stop"],
                }
            )
            if reason == "stop":
                cooldown_until[ticker] = ets + pd.Timedelta(minutes=p.cooldown_min)

        # entries
        if minute < or_end_min and p.require_or_complete:
            equity_marks.append((ts, cash))
            continue
        if minute >= flatten_min:
            equity_marks.append((ts, cash))
            continue
        open_risk = sum(positions[t]["risk"] for t in positions)
        for ticker, (lev, df) in frames.items():
            if ticker in positions:
                continue
            if ts not in df.index:
                continue
            if p.no_overnight and _last_session_bar(df, ts):
                continue
            cd = cooldown_until.get(ticker)
            if cd is not None and ts < cd:
                continue
            if len(positions) >= p.max_positions:
                break
            if open_risk >= ASSUMED_EQUITY * MAX_PORTFOLIO_RISK_PCT:
                break
            bar = df.loc[ts]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            atr14 = bar.get("atr14", np.nan)
            close = bar.get("close", np.nan)
            if not np.isfinite(atr14) or not np.isfinite(close) or atr14 <= 0:
                continue
            rg = regime_of(bar)
            side = pick_side(bar, p)
            if side is None:
                continue
            if open_risk + ASSUMED_EQUITY * p.risk_pct > ASSUMED_EQUITY * MAX_PORTFOLIO_RISK_PCT + 1e-9:
                continue
            shares, stop_dist, risk, notional = _size_position(float(close), float(atr14), lev, p)
            entry_cost = _cost(notional, p.slippage_bps)
            sign = 1 if side == "long" else -1
            if p.use_stops:
                stop = float(close) - sign * stop_dist
                target = float(close) + sign * 2.0 * stop_dist
            else:
                stop = np.nan
                target = np.nan
            positions[ticker] = {
                "side": side,
                "sign": sign,
                "entry_time": ts,
                "entry_px": float(close),
                "shares": shares,
                "notional": notional,
                "stop": stop,
                "target": target,
                "risk": risk,
                "entry_cost": entry_cost,
                "regime": rg,
            }
            open_risk += risk
        equity_marks.append((ts, cash))

    # force flatten remainder
    if positions:
        last_ts = index[-1]
        for ticker, pos in list(positions.items()):
            lev, df = frames[ticker]
            bar = last_bar_before(df, last_ts + pd.Timedelta(seconds=1))
            px = float(bar["close"]) if bar is not None else pos["entry_px"]
            slip = _cost(pos["notional"], p.slippage_bps)
            move = (px - pos["entry_px"]) * pos["shares"] * pos["sign"] * lev
            pnl = move - slip - pos["entry_cost"]
            cash += pnl
            trades.append(
                {
                    "ticker": ticker,
                    "leverage": lev,
                    "side": pos["side"],
                    "entry_time": pos["entry_time"],
                    "exit_time": last_ts,
                    "entry_px": pos["entry_px"],
                    "exit_px": px,
                    "shares": pos["shares"],
                    "notional": pos["notional"],
                    "pnl": pnl,
                    "return_pct": 100.0 * pnl / pos["notional"] if pos["notional"] else np.nan,
                    "reason": "forced_end",
                    "regime": pos["regime"],
                    "hold_minutes": (last_ts - pos["entry_time"]).total_seconds() / 60.0,
                    "stop": pos["stop"],
                }
            )

    tdf = pd.DataFrame(trades)
    eq = pd.Series({t: v for t, v in equity_marks}, dtype=float).sort_index()
    daily = eq.resample("1D").last().dropna()
    metrics = _metrics_from_trades(tdf, daily if len(daily) else eq)
    metrics["intervals"] = intervals
    metrics["params"] = asdict(p)
    return tdf, daily if len(daily) else eq, metrics


def _shared_rule_baseline(
    name: str,
    signal_fn,
    start=None,
    end=None,
    slippage_bps=SLIPPAGE_BPS_DEFAULT,
):
    """Portfolio baseline on the same universe/capital as the research engine."""
    p = Params(slippage_bps=slippage_bps, cooldown_min=0, max_hold_min=24 * 60)
    frames = {}
    for ticker, lev in UNIVERSE:
        _, df = load_underlying_frame(ticker)
        if df.empty:
            continue
        if start is not None:
            df = df[df.index >= _as_tz(start)]
        if end is not None:
            df = df[df.index < _as_tz(end)]
        frames[ticker] = (lev, df)
    if not frames:
        return pd.DataFrame(), pd.Series(dtype=float), {"n": 0, "name": name}

    index = None
    for _, df in frames.values():
        index = df.index if index is None else index.union(df.index)
    index = index.sort_values()
    cash = ASSUMED_EQUITY
    positions = {}
    trades = []
    marks = []
    flatten_min = RTH_END[0] * 60 + RTH_END[1] - FLAT_MINUTES_BEFORE_CLOSE

    for ts in index:
        minute = ts.hour * 60 + ts.minute
        flatten = minute >= flatten_min
        to_close = []
        for ticker, pos in list(positions.items()):
            lev, df = frames[ticker]
            if ts not in df.index:
                if flatten or ts.date() != pd.Timestamp(pos["entry_time"]).date():
                    forced = _force_eod_px(df, ts, pos)
                    if forced is not None:
                        px, ets = forced
                        to_close.append((ticker, px, "signal_or_eod", ets))
                continue
            bar = df.loc[ts]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            want = signal_fn(bar)
            if flatten or want != pos["side"] or _last_session_bar(df, ts):
                to_close.append((ticker, float(bar["close"]), "signal_or_eod", ts))
        for ticker, px, reason, ets in to_close:
            pos = positions.pop(ticker)
            lev = frames[ticker][0]
            notional = pos["notional"]
            pnl = lev * pos["shares"] * (px - pos["entry_px"]) * pos["sign"] - _cost(notional, slippage_bps) * 2
            cash += pnl
            trades.append(
                {
                    "ticker": ticker,
                    "side": pos["side"],
                    "entry_time": pos["entry_time"],
                    "exit_time": ets,
                    "pnl": pnl,
                    "reason": reason,
                }
            )
        if flatten:
            marks.append((ts, cash))
            continue
        open_risk = sum(positions[t]["risk"] for t in positions)
        for ticker, (lev, df) in frames.items():
            if ticker in positions or ts not in df.index:
                continue
            if _last_session_bar(df, ts):
                continue
            if len(positions) >= p.max_positions:
                break
            if open_risk >= ASSUMED_EQUITY * MAX_PORTFOLIO_RISK_PCT:
                break
            bar = df.loc[ts]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]
            side = signal_fn(bar)
            if side not in {"long", "short"}:
                continue
            atr14 = bar.get("atr14", np.nan)
            close = bar.get("close", np.nan)
            if not np.isfinite(atr14) or not np.isfinite(close) or atr14 <= 0:
                continue
            shares, stop_dist, risk, notional = _size_position(float(close), float(atr14), lev, p)
            if open_risk + risk > ASSUMED_EQUITY * MAX_PORTFOLIO_RISK_PCT + 1e-9:
                continue
            positions[ticker] = {
                "side": side,
                "sign": 1 if side == "long" else -1,
                "entry_time": ts,
                "entry_px": float(close),
                "shares": shares,
                "notional": notional,
                "risk": risk,
            }
            open_risk += risk
        marks.append((ts, cash))

    tdf = pd.DataFrame(trades)
    eq = pd.Series({t: v for t, v in marks}, dtype=float).sort_index().resample("1D").last().dropna()
    return tdf, eq, {**_metrics_from_trades(tdf, eq), "name": name}


def ema_crossover_baseline(start=None, end=None, slippage_bps=SLIPPAGE_BPS_DEFAULT):
    def signal(bar):
        e9, e20 = bar.get("ema9", np.nan), bar.get("ema20", np.nan)
        if not np.isfinite(e9) or not np.isfinite(e20):
            return None
        if e9 > e20:
            return "long"
        return None

    return _shared_rule_baseline("ema_crossover", signal, start, end, slippage_bps)


def vwap_momentum_baseline(start=None, end=None, slippage_bps=SLIPPAGE_BPS_DEFAULT):
    def signal(bar):
        c, v = bar.get("close", np.nan), bar.get("session_vwap", np.nan)
        if not np.isfinite(c) or not np.isfinite(v):
            return None
        if c > v:
            return "long"
        return None

    return _shared_rule_baseline("vwap_momentum", signal, start, end, slippage_bps)


def buy_hold(ticker: str, start, end, leverage: float = 1.0) -> dict:
    df = load_bars(ticker, "1d")
    if df.empty:
        return {"ticker": ticker, "error": "no_daily_data"}
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize(TZ)
    else:
        start = start.tz_convert(TZ)
    if end.tzinfo is None:
        end = end.tz_localize(TZ)
    else:
        end = end.tz_convert(TZ)
    w = df[(df.index >= start.normalize()) & (df.index <= end.normalize() + pd.Timedelta(days=1))]
    if len(w) < 2:
        return {"ticker": ticker, "error": "insufficient_range"}
    px0 = float(w.iloc[0]["close"])
    px1 = float(w.iloc[-1]["close"])
    ret = (px1 / px0 - 1.0) * leverage
    pnl = ASSUMED_EQUITY * ret
    eq = (w["close"] / px0) * ASSUMED_EQUITY
    if leverage != 1:
        # crude daily  leveraged path from close-to-close
        r = w["close"].pct_change().fillna(0) * leverage
        eq = ASSUMED_EQUITY * (1 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    rets = eq.pct_change().dropna()
    sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else np.nan
    return {
        "ticker": ticker,
        "leverage": leverage,
        "start": str(w.index[0]),
        "end": str(w.index[-1]),
        "total_pnl": float(eq.iloc[-1] - ASSUMED_EQUITY),
        "return_pct": float((eq.iloc[-1] / ASSUMED_EQUITY - 1) * 100),
        "max_drawdown_pct": float(dd),
        "sharpe": sharpe,
        "n": 1,
    }


def random_entry_baseline(template: pd.DataFrame, p: Params, start, end) -> tuple[pd.DataFrame, dict]:
    """Same count / same exits style: random RTH entries, identical stop/hold engine."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 0 if template is None or template.empty else len(template)
    frames = {}
    for ticker, lev in UNIVERSE:
        _, df = load_underlying_frame(ticker)
        if df.empty:
            continue
        df = df[(df.index >= _as_tz(start)) & (df.index < _as_tz(end))]
        frames[ticker] = (lev, df)
    if not frames or n == 0:
        return pd.DataFrame(), {"n": 0}
    keys = list(frames.keys())
    trades = []
    for _ in range(n):
        ticker = keys[int(rng.integers(0, len(keys)))]
        lev, df = frames[ticker]
        if df.empty:
            continue
        i = int(rng.integers(0, len(df)))
        bar = df.iloc[i]
        ts = df.index[i]
        minute = ts.hour * 60 + ts.minute
        flatten_min = RTH_END[0] * 60 + RTH_END[1] - p.flatten_before_close_min
        if minute >= flatten_min or _last_session_bar(df, ts):
            continue
        atr14 = bar.get("atr14", np.nan)
        close = bar.get("close", np.nan)
        if not np.isfinite(atr14) or atr14 <= 0:
            continue
        side = "long" if rng.random() < 0.5 else "short"
        sign = 1 if side == "long" else -1
        shares, stop_dist, risk, notional = _size_position(float(close), float(atr14), lev, p)
        stop = float(close) - sign * stop_dist
        target = float(close) + sign * 2 * stop_dist
        window = df.iloc[i : i + 200]
        exit_px = float(window.iloc[-1]["close"])
        reason = "time"
        exit_time = window.index[-1]
        held = 0.0
        for ets, b in window.iloc[1:].iterrows():
            held = (ets - ts).total_seconds() / 60.0
            hi, lo, cl = float(b["high"]), float(b["low"]), float(b["close"])
            minute = ets.hour * 60 + ets.minute
            if side == "long" and lo <= stop:
                exit_px, reason, exit_time = stop, "stop", ets
                break
            if side == "short" and hi >= stop:
                exit_px, reason, exit_time = stop, "stop", ets
                break
            if side == "long" and hi >= target:
                exit_px, reason, exit_time = target, "target", ets
                break
            if side == "short" and lo <= target:
                exit_px, reason, exit_time = target, "target", ets
                break
            if held >= p.max_hold_min or minute >= (RTH_END[0] * 60 + RTH_END[1] - p.flatten_before_close_min):
                exit_px, reason, exit_time = cl, "time_or_eod", ets
                break
        notional = shares * float(close)
        pnl = lev * shares * (exit_px - float(close)) * sign - _cost(notional, p.slippage_bps) * 2
        trades.append({"ticker": ticker, "side": side, "pnl": pnl, "reason": reason, "entry_time": ts, "exit_time": exit_time})
    tdf = pd.DataFrame(trades)
    dummy_eq = pd.Series([ASSUMED_EQUITY, ASSUMED_EQUITY + tdf["pnl"].sum() if not tdf.empty else ASSUMED_EQUITY])
    return tdf, {**_metrics_from_trades(tdf, dummy_eq), "name": "random_entry"}


def random_exit_baseline(strategy_trades: pd.DataFrame, p: Params) -> dict:
    """Keep research-engine entries; exit at a random later RTH bar within the max hold."""
    if strategy_trades is None or strategy_trades.empty:
        return {"n": 0, "name": "random_exit"}
    rng = np.random.default_rng(RANDOM_SEED + 1)
    frames = {t: load_underlying_frame(t)[1] for t, _ in UNIVERSE}
    pnls = []
    for _, tr in strategy_trades.iterrows():
        ticker = tr["ticker"]
        df = frames.get(ticker)
        if df is None or df.empty:
            continue
        entry = pd.Timestamp(tr["entry_time"])
        i = df.index.searchsorted(entry, side="left")
        if i >= len(df):
            continue
        end_ts = entry + pd.Timedelta(minutes=p.max_hold_min)
        if p.no_overnight:
            eod = entry.normalize() + pd.Timedelta(
                hours=RTH_END[0], minutes=RTH_END[1] - p.flatten_before_close_min
            )
            if end_ts > eod:
                end_ts = eod
        j = df.index.searchsorted(end_ts, side="right")
        window = df.iloc[i + 1 : max(i + 2, j)]
        if window.empty:
            continue
        k = int(rng.integers(0, len(window)))
        px = float(window.iloc[k]["close"])
        sign = 1 if tr["side"] == "long" else -1
        lev = float(tr.get("leverage", 2.0))
        shares = float(tr["shares"])
        notional = float(tr["notional"])
        pnl = lev * shares * (px - float(tr["entry_px"])) * sign - _cost(notional, p.slippage_bps) * 2
        pnls.append(pnl)
    tdf = pd.DataFrame({"pnl": pnls})
    dummy = pd.Series([ASSUMED_EQUITY, ASSUMED_EQUITY + (tdf["pnl"].sum() if len(tdf) else 0.0)])
    return {**_metrics_from_trades(tdf, dummy), "name": "random_exit"}


def user_equity(trips: pd.DataFrame) -> tuple[pd.Series, dict]:
    df = trips[trips["pnl"].notna()].sort_values("exit_time")
    eq = ASSUMED_EQUITY + df.set_index("exit_time")["pnl"].cumsum()
    daily = eq.resample("1D").last().ffill()
    return daily, {**_metrics_from_trades(df.rename(columns={"pnl": "pnl"}), daily), "name": "user_actual"}


def evaluate_variants(base: Params, start, cut, end) -> pd.DataFrame:
    """Book-motivated variants on a frozen cut. No nested winner-picking."""
    bp = {k: v for k, v in asdict(base).items() if k not in {"style", "use_stops", "tickers"}}
    specs = [
        ("trend_all_stops", Params(**bp, style="trend", use_stops=True)),
        ("trend_mu_nvda", Params(**bp, style="trend", use_stops=True, tickers=("MU", "NVDA"))),
        ("trend_nostop", Params(**bp, style="trend", use_stops=False)),
        ("trend_mu_nvda_nostop", Params(**bp, style="trend", use_stops=False, tickers=("MU", "NVDA"))),
        ("fade_all", Params(**bp, style="fade", use_stops=True)),
        ("fade_mu_nvda", Params(**bp, style="fade", use_stops=True, tickers=("MU", "NVDA"))),
        ("fade_nostop", Params(**bp, style="fade", use_stops=False)),
    ]
    rows = []
    for name, p in specs:
        print(f"    variant {name}")
        for split, a, b in (("IS", start, cut), ("OOS", cut, end)):
            tr, _, met = simulate(p, start=a, end=b)
            rnd_x = random_exit_baseline(tr, p)
            rows.append(
                {
                    "variant": name,
                    "split": split,
                    "style": p.style,
                    "use_stops": p.use_stops,
                    "tickers": ",".join(p.tickers) if p.tickers else "MU,NVDA,SOXX",
                    **{k: v for k, v in met.items() if k not in {"params", "intervals"}},
                    "random_exit_pnl": rnd_x.get("total_pnl"),
                    "beats_own_random_exit": bool(
                        met.get("n", 0) > 5 and met.get("total_pnl", 0) > (rnd_x.get("total_pnl") or 0)
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "strategy_variants.csv", index=False)
    return out


def sensitivity_and_wf(start, end) -> dict:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    grid = []
    span = _as_tz(end) - _as_tz(start)
    cut = _as_tz(start) + span * (1 - WALK_FORWARD_OOS_FRAC)
    best = None
    best_ev = -np.inf
    for stop in (1.0, 1.5, 2.0):
        for hold in (45, 90, 180):
            for cd in (0, 30):
                p = Params(stop_atr=stop, max_hold_min=hold, cooldown_min=cd)
                tr, eq, met = simulate(p, start=start, end=cut)
                print(
                    f"    grid stop={stop} hold={hold} cd={cd} n={met.get('n')} ev={met.get('expectancy')} pnl={met.get('total_pnl')}"
                )
                row = {
                    "stop_atr": stop,
                    "max_hold_min": hold,
                    "cooldown_min": cd,
                    "split": "IS",
                    **{k: v for k, v in met.items() if k not in {"params", "intervals"}},
                }
                grid.append(row)
                ev = met.get("expectancy") if met.get("n", 0) >= 10 else -np.inf
                if ev is not None and np.isfinite(ev) and ev > best_ev:
                    best_ev = ev
                    best = p
    if best is None:
        best = Params()
    tr_is, eq_is, met_is = simulate(best, start=start, end=cut)
    tr_oos, eq_oos, met_oos = simulate(best, start=cut, end=end)
    tr_all, eq_all, met_all = simulate(best, start=start, end=end)
    grid.append({"split": "OOS_selected", **{k: v for k, v in met_oos.items() if k not in {"params", "intervals"}}})
    gdf = pd.DataFrame(grid)
    gdf.to_csv(TABLE_DIR / "strategy_param_grid.csv", index=False)
    if not tr_all.empty:
        tr_all.to_csv(TABLE_DIR / "strategy_trade_log.csv", index=False)
    slip_rows = []
    for bps in SLIPPAGE_BPS_GRID:
        p = Params(**{**asdict(best), "slippage_bps": bps})
        _, _, m = simulate(p, start=cut, end=end)
        slip_rows.append({"slippage_bps": bps, "split": "OOS", **{k: v for k, v in m.items() if k not in {"params", "intervals"}}})
    pd.DataFrame(slip_rows).to_csv(TABLE_DIR / "strategy_slippage_sensitivity.csv", index=False)
    variants = evaluate_variants(best, start, cut, end)
    return {
        "best_params": asdict(best),
        "is": met_is,
        "oos": met_oos,
        "all": met_all,
        "trades_all": tr_all,
        "trades_oos": tr_oos,
        "equity_all": eq_all,
        "equity_oos": eq_oos,
        "cut": str(cut),
        "variants": variants,
    }


def run_backtests(trips: pd.DataFrame) -> dict:
    closed = trips[trips["pnl"].notna()]
    start = closed["entry_time"].min()
    end = closed["exit_time"].max() + pd.Timedelta(days=1)
    user_eq, user_met = user_equity(closed)
    wf = sensitivity_and_wf(start, end)
    ema_tr, ema_eq, ema_met = ema_crossover_baseline(start, end)
    vwap_tr, vwap_eq, vwap_met = vwap_momentum_baseline(start, end)
    cut = wf["cut"]
    _, _, ema_oos = ema_crossover_baseline(cut, end)
    _, _, vwap_oos = vwap_momentum_baseline(cut, end)
    p_best = Params(**wf["best_params"])
    rnd_tr, rnd_met = random_entry_baseline(wf["trades_all"], p_best, start, end)
    rnd_oos_tr, rnd_oos_met = random_entry_baseline(wf.get("trades_oos", wf["trades_all"]), p_best, cut, end)
    rnd_x = random_exit_baseline(wf["trades_all"], p_best)
    rnd_x_oos = random_exit_baseline(wf.get("trades_oos", wf["trades_all"]), p_best)
    bh = [
        buy_hold("MU", start, end, 1.0),
        buy_hold("MU", start, end, 2.0),
        buy_hold("NVDA", start, end, 1.0),
        buy_hold("SOXX", start, end, 1.0),
        buy_hold("SOXX", start, end, 3.0),
        buy_hold("MUU", start, end, 1.0),
        buy_hold("SOXL", start, end, 1.0),
        buy_hold("NVDL", start, end, 1.0),
    ]
    bh_oos = [
        buy_hold("MU", cut, end, 1.0),
        buy_hold("MU", cut, end, 2.0),
        buy_hold("NVDA", cut, end, 1.0),
        buy_hold("MUU", cut, end, 1.0),
        buy_hold("SOXL", cut, end, 1.0),
    ]
    pd.DataFrame(bh).to_csv(TABLE_DIR / "benchmark_buyhold.csv", index=False)
    pd.DataFrame(bh_oos).to_csv(TABLE_DIR / "benchmark_buyhold_oos.csv", index=False)
    compare = pd.DataFrame(
        [
            {"name": "user_actual", **{k: v for k, v in user_met.items() if k != "name"}},
            {"name": "strategy_all", **{k: v for k, v in wf["all"].items() if k not in {"params", "intervals"}}},
            {"name": "strategy_oos", **{k: v for k, v in wf["oos"].items() if k not in {"params", "intervals"}}},
            {"name": "ema_crossover_full", **{k: v for k, v in ema_met.items() if k != "name"}},
            {"name": "ema_crossover_oos", **{k: v for k, v in ema_oos.items() if k != "name"}},
            {"name": "vwap_momentum_full", **{k: v for k, v in vwap_met.items() if k != "name"}},
            {"name": "vwap_momentum_oos", **{k: v for k, v in vwap_oos.items() if k != "name"}},
            {"name": "random_entry", **{k: v for k, v in rnd_met.items() if k != "name"}},
            {"name": "random_entry_oos", **{k: v for k, v in rnd_oos_met.items() if k != "name"}},
            {"name": "random_exit", **{k: v for k, v in rnd_x.items() if k != "name"}},
            {"name": "random_exit_oos", **{k: v for k, v in rnd_x_oos.items() if k != "name"}},
        ]
    )
    compare.to_csv(TABLE_DIR / "benchmark_compare.csv", index=False)
    payload = {
        "user": user_met,
        "user_equity": user_eq,
        "wf": {k: v for k, v in wf.items() if k not in {"trades_all", "trades_oos", "equity_all", "equity_oos", "variants"}},
        "compare": compare,
        "buyhold": bh,
        "buyhold_oos": bh_oos,
        "strategy_trades": wf["trades_all"],
        "strategy_equity": wf["equity_all"],
        "variants": wf.get("variants", pd.DataFrame()),
        "start": str(start),
        "end": str(end),
        "cut": wf["cut"],
    }
    (PROC_DIR / "backtest_summary.json").write_text(
        json.dumps(
            {
                "best_params": wf["best_params"],
                "oos": {k: v for k, v in wf["oos"].items() if k not in {"params", "intervals"}},
                "compare": compare.to_dict(orient="records"),
                "buyhold": bh,
                "buyhold_oos": bh_oos,
                "cut": wf["cut"],
            },
            indent=2,
            default=str,
        )
    )
    return payload
