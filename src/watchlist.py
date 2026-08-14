"""Same-session volatility board. Ranks where the tape is moving — not a buy list."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from .config import REPORT_DIR, ROOT, TABLE_DIR, TZ, WATCHLIST, WATCHLIST_SKIP

MIN_HISTORY_DAYS = 60
EARNINGS_WARN_DAYS = 2
SLEEP_S = 0.25


def _hist(ticker: str, period: str, interval: str, prepost: bool = False) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
        prepost=prepost,
        timeout=60,
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


def _next_earnings(ticker: str) -> tuple[pd.Timestamp | None, str]:
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


def score_name(row: dict) -> dict:
    ticker = row["ticker"]
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
        "rvol_20": np.nan,
        "adv_20_musd": np.nan,
        "history_days": 0,
        "too_young": False,
        "next_earnings": "",
        "days_to_earnings": np.nan,
        "earnings_soon": False,
        "fuel_score": np.nan,
        "note": "",
    }
    if daily.empty or len(daily) < 15:
        out["error"] = "no_daily"
        out["note"] = "Yahoo returned no usable daily history."
        return out

    out["history_days"] = int(len(daily))
    out["too_young"] = len(daily) < MIN_HISTORY_DAYS
    last = float(daily["close"].iloc[-1])
    out["last"] = last
    if len(daily) >= 2:
        out["ret_1d_pct"] = float(daily["close"].iloc[-1] / daily["close"].iloc[-2] - 1) * 100
    if len(daily) >= 6:
        out["ret_5d_pct"] = float(daily["close"].iloc[-1] / daily["close"].iloc[-6] - 1) * 100
    rets = daily["close"].pct_change().dropna()
    if len(rets) >= 20:
        out["vol_20d_ann_pct"] = float(rets.tail(20).std() * np.sqrt(252) * 100)
    out["atr14_pct"] = _atr_pct(daily)
    if "volume" in daily.columns:
        dollar = daily["close"] * daily["volume"]
        out["adv_20_musd"] = float(dollar.tail(20).mean() / 1e6)

    today = pd.Timestamp.now(tz=TZ).normalize()
    if not intra.empty:
        sess = intra[intra.index.tz_convert(TZ).normalize() == today]
        if sess.empty:
            last_day = intra.index.tz_convert(TZ).normalize().max()
            sess = intra[intra.index.tz_convert(TZ).normalize() == last_day]
        if not sess.empty:
            hi, lo, cl = float(sess["high"].max()), float(sess["low"].min()), float(sess["close"].iloc[-1])
            if cl:
                out["today_range_pct"] = (hi - lo) / cl * 100
            if "volume" in sess.columns and "volume" in daily.columns and daily["volume"].tail(20).mean():
                # scale 5m session volume vs 20d average daily volume
                out["rvol_20"] = float(sess["volume"].sum() / daily["volume"].tail(20).mean())

    nxt, src = _next_earnings(ticker)
    if nxt is not None:
        out["next_earnings"] = str(nxt)
        days = (nxt.tz_convert(TZ).normalize() - today).days
        out["days_to_earnings"] = int(days)
        out["earnings_soon"] = 0 <= days <= EARNINGS_WARN_DAYS
        out["note"] = f"earnings source={src}"

    notes = []
    if out["too_young"]:
        notes.append(f"under {MIN_HISTORY_DAYS} daily bars — treat as young")
    if out["earnings_soon"]:
        notes.append("earnings within 2 sessions — flatten, do not hold through print")
    if row["group"] == "benchmark":
        notes.append("benchmark only")
    if row["ticker"] == "SOXX" or row["ticker"] == "SMH":
        notes.append("sector tape; do not jump to SOXL/SOXS overnight")
    if not row.get("ok_vehicle"):
        notes.append("no established 2x on this board — underlying same-session only, or skip")
    if notes:
        extra = "; ".join(notes)
        out["note"] = f"{out['note']}; {extra}" if out["note"] else extra

    # Fuel = how much the name typically moves. Not direction. Not a buy.
    parts = []
    if np.isfinite(out["vol_20d_ann_pct"]):
        parts.append(0.45 * min(out["vol_20d_ann_pct"] / 80.0, 1.5))
    if np.isfinite(out["atr14_pct"]):
        parts.append(0.35 * min(out["atr14_pct"] / 5.0, 1.5))
    if np.isfinite(out["today_range_pct"]):
        parts.append(0.20 * min(out["today_range_pct"] / 6.0, 1.5))
    out["fuel_score"] = float(sum(parts)) if parts else float("nan")
    return out


def build_watchlist() -> pd.DataFrame:
    rows = [score_name(item) for item in WATCHLIST]
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
        "**Not a buy list. Not a bot. Not live trading.** Ranked by how much the name typically moves (20-day vol, 14-day ATR, today's range). High score = more fuel for a same-day click, not a reason to buy.",
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
            "ret_1d_pct",
            "ret_5d_pct",
            "vol_20d_ann_pct",
            "atr14_pct",
            "today_range_pct",
            "rvol_20",
            "adv_20_musd",
            "fuel_score",
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
    lines.append("Yahoo Finance daily + 5-minute bars. Earnings dates from Yahoo when available. Empty cells mean the download failed, not that vol is zero.")
    lines.append("")
    md_path.write_text("\n".join(lines))
    payload = {
        "generated": now,
        "n": int(len(df)),
        "top": df.head(8)[["ticker", "fuel_score", "vol_20d_ann_pct", "today_range_pct"]].to_dict(
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


TAPE_NOTE = (
    "Memory has been the live tape (Sandisk / Western Digital / Micron). "
    "High fuel is a place to look, not a buy. Flatten before 16:00 ET."
)
