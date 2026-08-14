"""Same-session volatility board. Ranks where the tape is moving — not a buy list."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from .config import (
    REPORT_DIR,
    ROOT,
    TABLE_DIR,
    TZ,
    WATCHLIST,
    WATCHLIST_SKIP,
    et_session,
    expected_volume_frac,
)

MIN_HISTORY_DAYS = 60
EARNINGS_WARN_DAYS = 2
SLEEP_S = 0.08
YAHOO_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
# query1 is rate-limited from many IPs (HTTP 429). query2 still serves charts.
YAHOO_CHART = "https://query2.finance.yahoo.com/v8/finance/chart"


def _fetch_yahoo_json(url: str) -> dict:
    headers = {"User-Agent": YAHOO_UA, "Accept": "application/json"}
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            try:
                from curl_cffi import requests as cf_requests

                resp = cf_requests.get(url, impersonate="chrome", timeout=20, headers=headers)
                if resp.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    last_exc = RuntimeError(f"Yahoo HTTP {resp.status_code}")
                    time.sleep(0.4)
                    continue
                return resp.json()
            except ImportError:
                req = Request(url, headers=headers)
                with urlopen(req, timeout=20) as resp:
                    return json.loads(resp.read().decode())
        except HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_exc = exc
            time.sleep(0.5)
    if last_exc:
        raise last_exc
    raise RuntimeError("Yahoo chart empty")


def _yahoo_chart(ticker: str, interval: str, range_: str, prepost: bool = False) -> pd.DataFrame:
    params = {
        "interval": interval,
        "range": range_,
        "includePrePost": "true" if prepost else "false",
        "events": "div,splits",
    }
    url = f"{YAHOO_CHART}/{ticker}?{urlencode(params)}"
    try:
        payload = _fetch_yahoo_json(url)
    except Exception:
        return pd.DataFrame()
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return pd.DataFrame()
    node = result[0]
    ts = node.get("timestamp") or []
    quote = ((node.get("indicators") or {}).get("quote") or [{}])[0]
    if not ts:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(ts, unit="s", utc=True),
    )
    df.index = df.index.tz_convert(TZ)
    return df.dropna(subset=["close"])


def _hist_yf(ticker: str, period: str, interval: str, prepost: bool = False) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    t = yf.Ticker(ticker)
    df = t.history(
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
        prepost=prepost,
        timeout=20,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert(TZ)
    return df.dropna(subset=["close"])


def _hist(ticker: str, period: str, interval: str, prepost: bool = False) -> pd.DataFrame:
    df = _yahoo_chart(ticker, interval, period, prepost)
    if df.empty:
        df = _hist_yf(ticker, period, interval, prepost)
    return df


def _next_earnings(ticker: str) -> tuple[pd.Timestamp | None, str]:
    if yf is None:
        return None, "none"
    t = yf.Ticker(ticker)
    now = pd.Timestamp.now(tz=TZ)
    try:
        dates = t.get_earnings_dates(limit=12)
        if dates is not None and not dates.empty:
            idx = pd.to_datetime(dates.index)
            if getattr(idx, "tz", None) is None:
                idx = idx.tz_localize(TZ)
            else:
                idx = idx.tz_convert(TZ)
            future = idx[idx >= now.normalize()]
            if len(future):
                nxt = pd.Timestamp(future.min())
                return nxt, "yahoo_earnings_dates"
    except Exception:
        pass
    try:
        cal = t.calendar
        if cal is None:
            return None, "none"
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("earningsDate")
            if isinstance(raw, (list, tuple)) and raw:
                nxt = pd.Timestamp(raw[0])
            elif raw is not None:
                nxt = pd.Timestamp(raw)
            else:
                return None, "none"
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            nxt = pd.Timestamp(cal.loc["Earnings Date"].iloc[0])
        else:
            return None, "none"
        if nxt.tzinfo is None:
            nxt = nxt.tz_localize(TZ)
        else:
            nxt = nxt.tz_convert(TZ)
        return nxt, "yahoo_calendar"
    except Exception:
        return None, "none"


def _atr_pct(daily: pd.DataFrame, n: int = 14) -> float:
    if len(daily) < n + 1:
        return float("nan")
    prev = daily["close"].shift(1)
    tr = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prev).abs(),
            (daily["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(n).mean().iloc[-1]
    last = float(daily["close"].iloc[-1])
    return float(atr / last * 100) if last else float("nan")


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def _printed_move(range_pct, ret_pct) -> float:
    vals = [x for x in (range_pct, abs(ret_pct) if _finite(ret_pct) else np.nan) if _finite(x)]
    return max(vals) if vals else np.nan


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def _mins(idx: pd.DatetimeIndex) -> pd.Series:
    return idx.hour * 60 + idx.minute


def _slice_session(df: pd.DataFrame, day, start_min: int, end_min: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    day = pd.Timestamp(day)
    if day.tzinfo is None:
        day = day.tz_localize(TZ)
    day = day.tz_convert(TZ).normalize()
    part = df[df.index.normalize() == day]
    if part.empty:
        return part
    m = _mins(part.index)
    return part[(m >= start_min) & (m < end_min)]


def _sess_stats(bars: pd.DataFrame, ref: float | None) -> dict:
    out = {
        "last": np.nan,
        "open": np.nan,
        "high": np.nan,
        "low": np.nan,
        "range_pct": np.nan,
        "ret_pct": np.nan,
        "volume": np.nan,
        "vwap": np.nan,
        "n": 0,
    }
    if bars is None or bars.empty:
        return out
    bars = bars.dropna(subset=["close"])
    if bars.empty:
        return out
    last = float(bars["close"].iloc[-1])
    hi = float(bars["high"].max())
    lo = float(bars["low"].min())
    close_hi = float(bars["close"].max())
    close_lo = float(bars["close"].min())
    # After-hours often has one-tick spikes. If wicks dwarf the close path, use closes.
    if last and (hi - lo) > 3 * max(close_hi - close_lo, last * 0.002):
        hi, lo = close_hi, close_lo
    first = float(bars["open"].iloc[0]) if "open" in bars.columns and pd.notna(bars["open"].iloc[0]) else float(bars["close"].iloc[0])
    out.update({"last": last, "open": first, "high": hi, "low": lo, "n": int(len(bars))})
    if last:
        out["range_pct"] = (hi - lo) / last * 100
    if _finite(ref) and ref:
        out["ret_pct"] = (last / float(ref) - 1) * 100
    if "volume" in bars.columns:
        vol = float(bars["volume"].fillna(0).sum())
        out["volume"] = vol
        if vol > 0:
            typical = (bars["high"] + bars["low"] + bars["close"]) / 3
            out["vwap"] = float((typical * bars["volume"].fillna(0)).sum() / vol)
        else:
            out["vwap"] = float(bars["close"].mean())
    return out


def _prior_close(daily: pd.DataFrame, now: pd.Timestamp) -> float:
    last_idx = daily.index[-1]
    if last_idx.tz_convert(TZ).normalize() >= now.normalize() and len(daily) >= 2:
        return float(daily["close"].iloc[-2])
    return float(daily["close"].iloc[-1])


def _tape_lean(
    *,
    session: str,
    prior: float,
    last: float,
    atr: float,
    pm: dict,
    rth: dict,
    ah: dict,
    vwap: float,
    night_from: str,
) -> tuple[str, str, int, str]:
    """Tape read only. Fade = mixed. Not a forecast — the book had no directional edge."""
    thresh = max(0.25, 0.08 * atr) if _finite(atr) and atr > 0 else 0.25

    def sgn(x) -> int:
        if not _finite(x) or abs(float(x)) < thresh:
            return 0
        return 1 if float(x) > 0 else -1

    pm_x = pm.get("ret_pct")
    rth_x = rth.get("ret_pct")
    ah_x = ah.get("ret_pct")
    day_x = (last / prior - 1) * 100 if _finite(last) and _finite(prior) and prior else np.nan
    pm_s, rth_s, ah_s, day_s = sgn(pm_x), sgn(rth_x), sgn(ah_x), sgn(day_x)

    if session == "premarket":
        live, live_x, live_lbl = pm_s, pm_x, "PM"
    elif session == "rth":
        live, live_x, live_lbl = (rth_s or day_s), (rth_x if _finite(rth_x) else day_x), "cash"
    elif session == "afterhours":
        live, live_x, live_lbl = ah_s, ah_x, "night"
    else:
        live, live_x, live_lbl = day_s, day_x, "last"

    setup = ""
    if session == "premarket" and pm_s and ah_s:
        setup = "continuation" if pm_s == ah_s else "fade"
    elif session == "rth" and rth_s and pm_s:
        setup = "continuation" if rth_s == pm_s else "fade"
    elif session == "afterhours" and ah_s and rth_s:
        setup = "continuation" if ah_s == rth_s else "fade"
    elif session in {"overnight", "weekend"} and day_s and ah_s:
        setup = "continuation" if day_s == ah_s else "fade"

    if setup == "fade":
        why = "fade"
        if _finite(pm_x) and _finite(ah_x) and session == "premarket":
            why = f"fade · night {ah_x:+.1f}% vs PM {pm_x:+.1f}%"
        elif _finite(rth_x) and _finite(pm_x):
            why = f"fade · PM {pm_x:+.1f}% vs cash {rth_x:+.1f}%"
        return "mixed", why, 0, setup

    if live == 0:
        return "flat", "no lean yet", 0, setup

    lean = "up" if live > 0 else "down"
    bits = []
    if setup:
        bits.append(setup)
    if _finite(live_x):
        bits.append(f"{live_lbl} {live_x:+.1f}%")
    if _finite(last) and _finite(vwap) and vwap:
        vs = (last / vwap - 1) * 100
        if abs(vs) >= thresh:
            bits.append("above VWAP" if vs > 0 else "below VWAP")
            if (vs > 0) != (live > 0):
                return "mixed", " · ".join(bits + ["VWAP disagrees"]), 0, setup or "mixed"

    mag = abs(float(live_x)) / atr if _finite(live_x) and _finite(atr) and atr else 0
    strength = 2 if mag >= 0.35 else 1
    return lean, " · ".join(bits) if bits else "tape read, not a signal", strength, setup


def _fuel_score(*, printed: float, atr: float, rvol: float) -> tuple[float, str, float, float, bool]:
    """
    0–10: how much of this name's typical day has already printed *today*.
    Last night is context, not fuel. No early-session boost — a 1% wiggle
    should not look like a 5% gap.
    """
    if not _finite(printed) or not _finite(atr) or atr <= 0.15:
        return float("nan"), "no tape yet", float("nan"), float("nan"), False
    used = max(0.0, printed / atr)
    left = max(0.0, 1.0 - used)
    core = 10.0 * math.tanh(used / 0.40)
    if _finite(rvol):
        core *= 0.90 + 0.10 * min(max(float(rvol), 0.0), 2.0)
    if atr < 1.8:
        core *= 0.50 + 0.50 * (atr / 1.8)
    fuel = _clip(core, 0, 10)
    spent = used >= 0.70
    notes = [f"{used:.1f}× ATR used"]
    if spent:
        notes.append("spent — little left vs a normal day")
    elif left >= 0.45:
        notes.append(f"{left:.1f} ATR left")
    if fuel < 2.5:
        notes = ["quiet"] + [n for n in notes if "used" in n]
    return float(fuel), " · ".join(notes[:2]), float(used), float(left), spent


def _reuse_earnings(out: dict, prev: dict | None) -> None:
    raw = (prev or {}).get("next_earnings")
    if not raw:
        return
    try:
        nxt = pd.Timestamp(raw)
        if nxt.tzinfo is None:
            nxt = nxt.tz_localize(TZ)
        else:
            nxt = nxt.tz_convert(TZ)
        today = pd.Timestamp.now(tz=TZ).normalize()
        days = (nxt.normalize() - today).days
        out["next_earnings"] = str(nxt)
        out["days_to_earnings"] = int(days)
        out["earnings_soon"] = 0 <= days <= EARNINGS_WARN_DAYS
    except (TypeError, ValueError):
        return


def score_name(row: dict, prev: dict | None = None) -> dict:
    ticker = row["ticker"]
    now = pd.Timestamp.now(tz=TZ)
    sess = et_session(now.to_pydatetime())
    daily = _hist(ticker, "6mo", "1d")
    intra = _hist(ticker, "5d", "5m", prepost=True)
    time.sleep(SLEEP_S)

    out = {
        **row,
        "error": "",
        "last": np.nan,
        "ret_1d_pct": np.nan,
        "ret_5d_pct": np.nan,
        "vol_20d_ann_pct": np.nan,
        "atr14_pct": np.nan,
        "today_range_pct": np.nan,
        "pm_ret_pct": np.nan,
        "pm_range_pct": np.nan,
        "rth_ret_pct": np.nan,
        "rth_range_pct": np.nan,
        "ah_ret_pct": np.nan,
        "ah_range_pct": np.nan,
        "night_from": "",
        "gap_pct": np.nan,
        "vwap": np.nan,
        "rvol_20": np.nan,
        "adv_20_musd": np.nan,
        "history_days": 0,
        "too_young": False,
        "next_earnings": "",
        "days_to_earnings": np.nan,
        "earnings_soon": False,
        "fuel_score": np.nan,
        "fuel_note": "",
        "lean": "flat",
        "lean_why": "",
        "lean_strength": 0,
        "setup": "",
        "atr_used": np.nan,
        "atr_left": np.nan,
        "spent": False,
        "split_pm": 0.0,
        "split_rth": 0.0,
        "split_ah": 0.0,
        "session": sess["kind"],
        "note": "",
    }
    if daily.empty or len(daily) < 15:
        out["error"] = "no_daily"
        out["note"] = "Yahoo returned no usable daily history."
        return out

    out["history_days"] = int(len(daily))
    out["too_young"] = len(daily) < MIN_HISTORY_DAYS
    prior = _prior_close(daily, now)
    out["atr14_pct"] = _atr_pct(daily)
    rets = daily["close"].pct_change().dropna()
    if len(rets) >= 20:
        out["vol_20d_ann_pct"] = float(rets.tail(20).std() * np.sqrt(252) * 100)
    if len(daily) >= 6:
        out["ret_5d_pct"] = float(daily["close"].iloc[-1] / daily["close"].iloc[-6] - 1) * 100
    adv_shares = np.nan
    if "volume" in daily.columns:
        dollar = daily["close"] * daily["volume"]
        out["adv_20_musd"] = float(dollar.tail(20).mean() / 1e6)
        adv_shares = float(daily["volume"].tail(20).mean())

    pre_m = 4 * 60
    rth_m = 9 * 60 + 30
    close_m = 16 * 60
    ah_m = 20 * 60

    pm = _sess_stats(pd.DataFrame(), prior)
    rth = _sess_stats(pd.DataFrame(), prior)
    ah = _sess_stats(pd.DataFrame(), prior)
    active_day = now.normalize()
    complete_day = sess["kind"] in {"overnight", "weekend"}
    if not intra.empty:
        days = sorted(intra.index.normalize().unique())
        if active_day not in days:
            active_day = days[-1]
            complete_day = True
        pm = _sess_stats(_slice_session(intra, active_day, pre_m, rth_m), prior)
        rth_bars = _slice_session(intra, active_day, rth_m, close_m)
        rth = _sess_stats(rth_bars, prior)
        ah_bars = _slice_session(intra, active_day, close_m, ah_m)
        night_from = "ah"
        prior_days = [d for d in days if d < active_day]
        if ah_bars.empty and prior_days:
            ah_bars = _slice_session(intra, prior_days[-1], close_m, ah_m)
            night_from = "last_night"
        if night_from == "ah":
            night_ref = float(rth_bars["close"].iloc[-1]) if not rth_bars.empty else prior
        elif prior_days:
            prev_rth = _slice_session(intra, prior_days[-1], rth_m, close_m)
            night_ref = float(prev_rth["close"].iloc[-1]) if not prev_rth.empty else prior
        else:
            night_ref = prior
        ah = _sess_stats(ah_bars, night_ref)
        out["night_from"] = night_from if not ah_bars.empty else ""

    out["pm_ret_pct"] = pm["ret_pct"]
    out["pm_range_pct"] = pm["range_pct"]
    out["rth_ret_pct"] = rth["ret_pct"]
    out["rth_range_pct"] = rth["range_pct"]
    out["ah_ret_pct"] = ah["ret_pct"]
    out["ah_range_pct"] = ah["range_pct"]
    if _finite(pm.get("open")) and prior:
        out["gap_pct"] = (float(pm["open"]) / prior - 1) * 100
    elif _finite(rth.get("open")) and prior:
        out["gap_pct"] = (float(rth["open"]) / prior - 1) * 100

    last_candidates = [pm["last"], rth["last"], ah["last"] if out.get("night_from") == "ah" else np.nan]
    last = next((float(x) for x in reversed(last_candidates) if _finite(x)), np.nan)
    if not _finite(last):
        last = float(daily["close"].iloc[-1])
    out["last"] = last
    if prior:
        out["ret_1d_pct"] = (last / prior - 1) * 100

    highs = [x for x in (pm["high"], rth["high"], ah["high"] if out.get("night_from") == "ah" else np.nan) if _finite(x)]
    lows = [x for x in (pm["low"], rth["low"], ah["low"] if out.get("night_from") == "ah" else np.nan) if _finite(x)]
    if highs and lows and last:
        out["today_range_pct"] = (max(highs) - min(lows)) / last * 100
    elif last and {"high", "low"}.issubset(daily.columns):
        hi, lo = float(daily["high"].iloc[-1]), float(daily["low"].iloc[-1])
        out["today_range_pct"] = (hi - lo) / last * 100

    vwap = np.nan
    if sess["kind"] == "premarket" and _finite(pm.get("vwap")):
        vwap = pm["vwap"]
    elif sess["kind"] in {"rth", "overnight", "weekend"} and _finite(rth.get("vwap")):
        vwap = rth["vwap"]
    elif sess["kind"] == "afterhours" and _finite(ah.get("vwap")):
        vwap = ah["vwap"]
    elif _finite(rth.get("vwap")):
        vwap = rth["vwap"]
    elif _finite(pm.get("vwap")):
        vwap = pm["vwap"]
    out["vwap"] = vwap

    vol_so_far = 0.0
    for block, use in ((pm, True), (rth, True), (ah, out.get("night_from") == "ah")):
        if use and _finite(block.get("volume")):
            vol_so_far += float(block["volume"])
    if vol_so_far and _finite(adv_shares) and adv_shares > 0:
        frac = expected_volume_frac(sess["kind"], int(sess["mins"]), complete=complete_day)
        out["rvol_20"] = vol_so_far / adv_shares / max(frac, 0.03)

    # Earnings dates go through Yahoo query1 (crumb), which 429s and hangs refresh.
    _reuse_earnings(out, prev)

    lean, why, strength, setup = _tape_lean(
        session=sess["kind"],
        prior=prior,
        last=last,
        atr=out["atr14_pct"],
        pm=pm,
        rth=rth,
        ah=ah,
        vwap=vwap,
        night_from=out.get("night_from") or "",
    )
    out["lean"] = lean
    out["lean_why"] = why
    out["lean_strength"] = strength
    out["setup"] = setup

    # Fuel is today only. Last night stays in the Night column for context.
    printed = _printed_move(out["today_range_pct"], out["ret_1d_pct"])
    printed = _printed_move(printed, out["pm_ret_pct"])
    printed = _printed_move(printed, out["rth_ret_pct"])
    if _finite(out["gap_pct"]):
        printed = _printed_move(printed, out["gap_pct"])
    if out.get("night_from") == "ah":
        printed = _printed_move(printed, out["ah_ret_pct"])
        printed = _printed_move(printed, out["ah_range_pct"])
    fuel, fuel_note, used, left, spent = _fuel_score(
        printed=printed,
        atr=out["atr14_pct"],
        rvol=out["rvol_20"],
    )
    out["fuel_score"] = fuel
    out["fuel_note"] = fuel_note
    out["atr_used"] = used
    out["atr_left"] = left
    out["spent"] = spent
    out["split_pm"] = abs(float(out["pm_ret_pct"])) if _finite(out["pm_ret_pct"]) else 0.0
    out["split_rth"] = abs(float(out["rth_ret_pct"])) if _finite(out["rth_ret_pct"]) else 0.0
    out["split_ah"] = (
        abs(float(out["ah_ret_pct"]))
        if out.get("night_from") == "ah" and _finite(out["ah_ret_pct"])
        else 0.0
    )

    notes = []
    if out["too_young"]:
        notes.append(f"under {MIN_HISTORY_DAYS} daily bars — treat as young")
    if _finite(out.get("days_to_earnings")) and int(out["days_to_earnings"]) == 0:
        notes.append("print today — do not hold through it")
    elif out["earnings_soon"]:
        notes.append("earnings within 2 sessions — flatten, do not hold through print")
    if row["group"] == "benchmark":
        notes.append("benchmark only")
    if row["ticker"] in {"SOXX", "SMH"}:
        notes.append("sector tape; do not jump to SOXL/SOXS overnight")
    if not row.get("ok_vehicle"):
        notes.append("no established 2x — underlying same-session only, or skip")
    if spent:
        notes.append("most of a normal day already printed — chasing leftover is how overnight got hurt")
    if notes:
        out["note"] = "; ".join(notes)
    return out


def build_watchlist() -> pd.DataFrame:
    prev = {r.get("ticker"): r for r in (load_cached_watchlist().get("rows") or []) if r.get("ticker")}
    rows = [score_name(item, prev.get(item["ticker"])) for item in WATCHLIST]
    df = pd.DataFrame(rows)
    df = df.sort_values(["too_young", "fuel_score"], ascending=[True, False], na_position="last")
    return df.reset_index(drop=True)


def write_watchlist(df: pd.DataFrame, tape_note: str = "") -> dict[str, str]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "watchlist.csv"
    md_path = REPORT_DIR / "WATCHLIST.md"
    df.to_csv(csv_path, index=False)

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        "# Same-session volatility board",
        "",
        "**Not a buy list. Not a bot.** Ranked by **fuel (0–10)**: how much of this name's typical day has already printed *today*. Last night is shown, not scored. **Lean** is a tape read. Fade = mixed. Not a forecast.",
        "",
        f"Generated: {now}",
        "",
        "## How to use this",
        "",
        "1. Look at the top **memory / NVIDIA / AMD** rows if you want the tape you already know.",
        "2. If you click, same session only. Flatten before 16:00 ET.",
        "3. Prefer the **vehicle** column only when `ok_vehicle` is true (established 2x). Otherwise trade the underlying or skip.",
        "4. If `earnings_soon` is true, do not hold through the print.",
        "5. Skip anything in the ban list below — those already cost this book.",
        "",
        "## Ban list (do not hunt these)",
        "",
    ]
    for sym, why in WATCHLIST_SKIP.items():
        lines.append(f"- `{sym}`: {why}")
    lines.append("")
    if tape_note:
        lines.append("## Tape context (public web, not a signal)")
        lines.append("")
        lines.append(tape_note)
        lines.append("")
    lines.append("## Ranked names")
    lines.append("")
    show = df[
        [
            "ticker",
            "name",
            "group",
            "vehicle",
            "ok_vehicle",
            "last",
            "lean",
            "lean_why",
            "fuel_score",
            "fuel_note",
            "atr_used",
            "atr_left",
            "spent",
            "setup",
            "pm_ret_pct",
            "pm_range_pct",
            "rth_ret_pct",
            "rth_range_pct",
            "ah_ret_pct",
            "ah_range_pct",
            "gap_pct",
            "ret_1d_pct",
            "atr14_pct",
            "today_range_pct",
            "rvol_20",
            "adv_20_musd",
            "days_to_earnings",
            "earnings_soon",
            "too_young",
            "note",
        ]
    ].copy()
    fmt = show.copy()
    for c in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[c]):
            fmt[c] = fmt[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    lines.append("| " + " | ".join(fmt.columns) + " |")
    lines.append("|" + "|".join(["---"] * len(fmt.columns)) + "|")
    for row in fmt.itertuples(index=False):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    lines.append("")
    lines.append("Yahoo Finance daily + 5-minute extended-hours bars (premarket 4:00–9:30 ET, cash 9:30–16:00, night 16:00–20:00). Lean is a description of the tape, not a buy/sell. Empty cells mean the download failed.")
    lines.append("")
    md_path.write_text("\n".join(lines))
    payload = {
        "generated": now,
        "n": int(len(df)),
        "top": df.head(8)[["ticker", "fuel_score", "lean", "pm_ret_pct", "ah_ret_pct", "today_range_pct"]].to_dict(
            orient="records"
        ),
    }
    (TABLE_DIR / "watchlist_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    recs = records(df)
    payload_board = {
        "generated": now,
        "rows": recs,
        "tape": tape_note or TAPE_NOTE,
        "ban": WATCHLIST_SKIP,
        "fuel_legend": FUEL_LEGEND,
    }
    (TABLE_DIR / "watchlist.json").write_text(json.dumps(payload_board, indent=2, default=_json_val))
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "watchlist.json").write_text(json.dumps(payload_board, indent=2, default=_json_val))
    return {"csv": str(csv_path), "md": str(md_path)}


def _json_val(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    out = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        out.append(clean)
    return out


def load_cached_watchlist() -> dict:
    path = TABLE_DIR / "watchlist.json"
    if path.exists():
        return json.loads(path.read_text())
    csv_path = TABLE_DIR / "watchlist.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return {"generated": None, "rows": records(df)}
    return {"generated": None, "rows": []}


FUEL_LEGEND = (
    "Fuel 0–10 = how much of a normal day has already printed today (gap + session vs ATR). "
    "Last night is context, not the rank. Lean is a tape read — fade comes back mixed. Not a forecast."
)

TAPE_NOTE = (
    "Look at high fuel. Lean is not a buy. If it says spent or print today, do not chase. "
    "Flatten 2x before 16:00 ET. Do not carry overnight."
)
