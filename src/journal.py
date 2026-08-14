"""Manual trade journal. Separate from the broker FIFO book."""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import pandas as pd

from .config import INSTRUMENTS, JOURNAL_PATH, PROC_DIR, TZ, WATCHLIST_SKIP, et_session

RTH_CLOSE_HOUR = 16


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ)


def et_clock() -> dict:
    now = _now()
    sess = et_session(now.to_pydatetime())
    return {
        "et": str(now),
        "et_local": sess["et_local"],
        "et_label": sess["et_label"],
        "session": sess["kind"],
        "after_close": sess["after_close"],
        "morning_grind": sess["morning_grind"],
        "premarket": sess["premarket"],
        "afterhours": sess["afterhours"],
        "weekend": sess["weekend"],
    }


def _parse_et(raw: str | None) -> pd.Timestamp | None:
    if not raw:
        return None
    t = pd.Timestamp(raw)
    if t.tzinfo is None:
        t = t.tz_localize(TZ)
    else:
        t = t.tz_convert(TZ)
    return t


def _load() -> list[dict]:
    if not JOURNAL_PATH.exists():
        return []
    data = json.loads(JOURNAL_PATH.read_text())
    return data.get("trades", [])


def _save(trades: list[dict]) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL_PATH.write_text(json.dumps({"trades": trades}, indent=2, default=str))


def _annotate(trade: dict, others: list[dict]) -> dict:
    t = dict(trade)
    sym = str(t.get("symbol", "")).upper()
    t["symbol"] = sym
    meta = INSTRUMENTS.get(sym, {})
    t["family"] = meta.get("family")
    t["leverage"] = meta.get("leverage")
    t["vehicle_known"] = bool(meta)
    entry = _parse_et(t.get("entry_time"))
    exit_t = _parse_et(t.get("exit_time"))
    qty = float(t.get("qty") or 0)
    px_in = float(t.get("entry_price") or 0)
    px_out = t.get("exit_price")
    side = str(t.get("side", "buy")).lower()
    sign = 1 if side == "buy" else -1
    t["entry_notional"] = qty * px_in
    t["status"] = "closed" if px_out not in (None, "") and exit_t is not None else "open"
    if t["status"] == "closed":
        px_out = float(px_out)
        t["pnl"] = (px_out - px_in) * qty * sign
        t["return_pct"] = ((px_out / px_in) - 1) * 100 * sign if px_in else None
        hold = (exit_t - entry).total_seconds() / 60.0 if entry is not None else None
        t["hold_minutes"] = hold
        overnight = False
        if entry is not None and exit_t is not None:
            overnight = entry.date() != exit_t.date() or (
                entry.hour < RTH_CLOSE_HOUR <= exit_t.hour and entry.date() == exit_t.date()
            )
        t["overnight"] = overnight
        t["same_session"] = not overnight
    else:
        t["pnl"] = None
        t["return_pct"] = None
        t["hold_minutes"] = None
        now = _now()
        overnight = False
        if entry is not None:
            overnight = entry.date() != now.date() or now.hour >= RTH_CLOSE_HOUR
        t["overnight"] = overnight
        t["same_session"] = not overnight

    warnings = []
    if sym in WATCHLIST_SKIP:
        warnings.append(WATCHLIST_SKIP[sym])
    if t.get("overnight"):
        warnings.append("Overnight 2x/3x is where this book lost about $1,100. Flatten before 16:00 ET.")
    if entry is not None and 10 <= entry.hour < 12:
        warnings.append("10:00–12:00 ET was the worst clock on the historical book (−$949).")
    opens_same = [
        o
        for o in others
        if o.get("id") != t.get("id")
        and str(o.get("symbol", "")).upper() == sym
        and o.get("status") == "open"
    ]
    if opens_same:
        warnings.append("Another open lot in this name. Averaging down campaigns were −$343.")
    t["warnings"] = warnings
    return t


def list_trades() -> list[dict]:
    raw = _load()
    annotated = []
    for t in raw:
        rest = [x for x in raw if x.get("id") != t.get("id")]
        annotated.append(_annotate(t, rest))
    annotated.sort(key=lambda r: r.get("entry_time") or "", reverse=True)
    return annotated


def stats() -> dict:
    trades = list_trades()
    closed = [t for t in trades if t["status"] == "closed" and t.get("pnl") is not None]
    opens = [t for t in trades if t["status"] == "open"]
    pnl = sum(t["pnl"] for t in closed)
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    same = [t for t in closed if t.get("same_session")]
    overnight = [t for t in closed if t.get("overnight")]
    return {
        "n_open": len(opens),
        "n_closed": len(closed),
        "open_notional": sum(float(t.get("entry_notional") or 0) for t in opens),
        "realized_pnl": pnl,
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "same_session_pnl": sum(t["pnl"] for t in same),
        "overnight_pnl": sum(t["pnl"] for t in overnight),
        "avg_win": (sum(t["pnl"] for t in wins) / len(wins)) if wins else None,
        "avg_loss": (sum(t["pnl"] for t in losses) / len(losses)) if losses else None,
    }


def add_trade(payload: dict) -> dict:
    trades = _load()
    entry = _parse_et(payload.get("entry_time")) or _now()
    trade = {
        "id": str(uuid.uuid4())[:8],
        "symbol": str(payload.get("symbol", "")).upper().strip(),
        "side": str(payload.get("side", "buy")).lower(),
        "qty": float(payload["qty"]),
        "entry_price": float(payload["entry_price"]),
        "entry_time": str(entry),
        "exit_price": None,
        "exit_time": None,
        "notes": str(payload.get("notes") or ""),
        "source": "journal",
        "created_at": str(_now()),
    }
    if not trade["symbol"]:
        raise ValueError("Symbol is required")
    trades.append(trade)
    _save(trades)
    return _annotate(trade, trades)


def close_trade(trade_id: str, payload: dict) -> dict:
    trades = _load()
    found = None
    for t in trades:
        if t["id"] == trade_id:
            found = t
            break
    if found is None:
        raise KeyError(trade_id)
    exit_t = _parse_et(payload.get("exit_time")) or _now()
    found["exit_price"] = float(payload["exit_price"])
    found["exit_time"] = str(exit_t)
    if payload.get("notes"):
        found["notes"] = str(payload["notes"])
    _save(trades)
    rest = [x for x in trades if x["id"] != trade_id]
    return _annotate(found, rest)


def update_trade(trade_id: str, payload: dict) -> dict:
    trades = _load()
    found = None
    for t in trades:
        if t["id"] == trade_id:
            found = t
            break
    if found is None:
        raise KeyError(trade_id)
    for key in ("symbol", "side", "qty", "entry_price", "entry_time", "exit_price", "exit_time", "notes"):
        if key in payload and payload[key] not in (None, ""):
            found[key] = payload[key]
    if "qty" in found:
        found["qty"] = float(found["qty"])
    if "entry_price" in found:
        found["entry_price"] = float(found["entry_price"])
    if found.get("entry_time"):
        found["entry_time"] = str(_parse_et(found["entry_time"]))
    if found.get("exit_time"):
        found["exit_time"] = str(_parse_et(found["exit_time"]))
        found["exit_price"] = float(found["exit_price"])
    _save(trades)
    rest = [x for x in trades if x["id"] != trade_id]
    return _annotate(found, rest)


def delete_trade(trade_id: str) -> None:
    trades = _load()
    kept = [t for t in trades if t["id"] != trade_id]
    if len(kept) == len(trades):
        raise KeyError(trade_id)
    _save(kept)


def seed_open_from_broker() -> list[dict]:
    """If the journal file does not exist yet, copy the current broker open lot."""
    if JOURNAL_PATH.exists():
        return list_trades()
    path = PROC_DIR / "open_lots.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    trades = []
    for _, row in df.iterrows():
        trades.append(
            {
                "id": str(uuid.uuid4())[:8],
                "symbol": str(row["symbol"]).upper(),
                "side": "buy",
                "qty": float(row["qty"]),
                "entry_price": float(row["entry_price"]),
                "entry_time": str(row["entry_time"]),
                "exit_price": None,
                "exit_time": None,
                "notes": "Imported from broker open inventory",
                "source": "broker_open",
                "created_at": str(_now()),
            }
        )
    _save(trades)
    return list_trades()


def broker_history(limit: int = 80) -> list[dict]:
    path = PROC_DIR / "round_trips.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df.sort_values("entry_time", ascending=False).head(limit)
    cols = [
        "lot_id",
        "symbol",
        "family",
        "qty",
        "entry_price",
        "exit_price",
        "entry_time",
        "exit_time",
        "overnight",
        "pnl",
        "return_pct",
        "hold_minutes",
        "entry_session",
        "outcome",
    ]
    cols = [c for c in cols if c in df.columns]
    rows = []
    for rec in df[cols].to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and pd.isna(v)):
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        rows.append(clean)
    return rows
