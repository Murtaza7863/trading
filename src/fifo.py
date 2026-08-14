"""FIFO lot matching, campaign reconstruction, and trade labels."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    CATASTROPHIC_LOSS_PCT,
    LARGE_WIN_PCT,
    PROC_DIR,
    SMALL_BAND_PCT,
)


@dataclass
class OpenLot:
    fill_id: int
    qty_remaining: float
    price: float
    time: pd.Timestamp
    side: str
    campaign_id: int


@dataclass
class Campaign:
    campaign_id: int
    symbol: str
    family: str
    direction: str  # long ETF inventory
    entries: list = field(default_factory=list)
    exits: list = field(default_factory=list)
    qty_open: float = 0.0
    open_time: pd.Timestamp | None = None
    close_time: pd.Timestamp | None = None


def _signed_qty(side: str, qty: float) -> float:
    return qty if side == "BUY" else -qty


def match_fifo(fills: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match fills FIFO per symbol.

    Round-trips are long-ETF inventory cycles (buy then sell). Inverse ETFs
    are still long the ETF shares; economic direction is stored separately.
    """
    lots: dict[str, deque[OpenLot]] = defaultdict(deque)
    campaigns: dict[str, Campaign | None] = defaultdict(lambda: None)
    next_campaign = 1
    round_trips = []
    campaign_rows = []

    fills = fills.sort_values(["filled_time", "fill_id"]).reset_index(drop=True)

    def close_campaign(sym: str) -> None:
        nonlocal campaign_rows
        camp = campaigns[sym]
        if camp is None:
            return
        camp.close_time = camp.exits[-1]["time"] if camp.exits else camp.open_time
        entry_qty = sum(e["qty"] for e in camp.entries)
        exit_qty = sum(x["qty"] for x in camp.exits)
        entry_notional = sum(e["qty"] * e["price"] for e in camp.entries)
        exit_notional = sum(x["qty"] * x["price"] for x in camp.exits)
        avg_entry = entry_notional / entry_qty if entry_qty else np.nan
        avg_exit = exit_notional / exit_qty if exit_qty else np.nan
        pnl = exit_notional - entry_notional
        ret = (avg_exit / avg_entry - 1.0) * 100.0 if avg_entry else np.nan
        n_adds = max(0, len(camp.entries) - 1)
        avg_down = False
        running_avg = None
        running_qty = 0.0
        for e in camp.entries:
            if running_qty > 0 and e["price"] < running_avg - 1e-9:
                avg_down = True
            running_avg = (
                (running_avg * running_qty + e["price"] * e["qty"]) / (running_qty + e["qty"])
                if running_qty > 0
                else e["price"]
            )
            running_qty += e["qty"]
        duration_min = (
            (camp.close_time - camp.open_time).total_seconds() / 60.0
            if camp.close_time is not None and camp.open_time is not None
            else np.nan
        )
        overnight = False
        if camp.open_time is not None and camp.close_time is not None:
            overnight = camp.open_time.date() != camp.close_time.date()
            if not overnight:
                close = camp.open_time.normalize() + pd.Timedelta(hours=16)
                overnight = camp.open_time < close <= camp.close_time
        closed = abs(camp.qty_open) < 1e-9
        campaign_rows.append(
            {
                "campaign_id": camp.campaign_id,
                "symbol": camp.symbol,
                "family": camp.family,
                "direction": camp.direction,
                "n_entries": len(camp.entries),
                "n_exits": len(camp.exits),
                "n_adds": n_adds,
                "averaged_down": avg_down,
                "qty": entry_qty,
                "avg_entry": avg_entry,
                "avg_exit": avg_exit if closed else np.nan,
                "entry_time": camp.open_time,
                "exit_time": camp.close_time if closed else pd.NaT,
                "hold_minutes": duration_min if closed else np.nan,
                "overnight": overnight if closed else False,
                "pnl": pnl if closed else np.nan,
                "return_pct": ret if closed else np.nan,
                "entry_notional": entry_notional,
                "status": "closed" if closed else "open",
            }
        )
        campaigns[sym] = None

    for _, fill in fills.iterrows():
        sym = fill["symbol"]
        side = fill["side"]
        qty = float(fill["qty"])
        px = float(fill["price"])
        ts = fill["filled_time"]
        remaining = qty

        if side == "BUY":
            camp = campaigns[sym]
            if camp is None:
                camp = Campaign(
                    campaign_id=next_campaign,
                    symbol=sym,
                    family=fill["family"],
                    direction="long",
                    open_time=ts,
                )
                next_campaign += 1
                campaigns[sym] = camp
            camp.entries.append(
                {"fill_id": fill["fill_id"], "qty": qty, "price": px, "time": ts}
            )
            camp.qty_open += qty
            lots[sym].append(
                OpenLot(
                    fill_id=int(fill["fill_id"]),
                    qty_remaining=qty,
                    price=px,
                    time=ts,
                    side="BUY",
                    campaign_id=camp.campaign_id,
                )
            )
            continue

        # SELL: close FIFO longs. Residual sell would be a short; none expected.
        while remaining > 1e-9 and lots[sym]:
            lot = lots[sym][0]
            take = min(lot.qty_remaining, remaining)
            pnl = (px - lot.price) * take
            ret = (px / lot.price - 1.0) * 100.0
            hold_min = (ts - lot.time).total_seconds() / 60.0
            overnight = lot.time.date() != ts.date()
            if not overnight:
                close = lot.time.normalize() + pd.Timedelta(hours=16)
                overnight = lot.time < close <= ts
            round_trips.append(
                {
                    "lot_id": len(round_trips) + 1,
                    "campaign_id": lot.campaign_id,
                    "symbol": sym,
                    "name": fill["name"],
                    "family": fill["family"],
                    "underlying": fill["underlying"],
                    "leverage": fill["leverage"],
                    "side_bias": fill["side_bias"],
                    "economic_side": fill["economic_side"],
                    "product": fill["product"],
                    "qty": take,
                    "entry_price": lot.price,
                    "exit_price": px,
                    "entry_time": lot.time,
                    "exit_time": ts,
                    "entry_fill_id": lot.fill_id,
                    "exit_fill_id": fill["fill_id"],
                    "hold_minutes": hold_min,
                    "overnight": overnight,
                    "intraday": (not overnight) and hold_min < 24 * 60,
                    "pnl": pnl,
                    "return_pct": ret,
                    "entry_notional": take * lot.price,
                    "exit_notional": take * px,
                }
            )
            lot.qty_remaining -= take
            remaining -= take
            camp = campaigns[sym]
            if camp is not None:
                camp.exits.append(
                    {"fill_id": fill["fill_id"], "qty": take, "price": px, "time": ts}
                )
                camp.qty_open -= take
            if lot.qty_remaining <= 1e-9:
                lots[sym].popleft()
            if camp is not None and camp.qty_open <= 1e-9:
                close_campaign(sym)

        if remaining > 1e-9:
            # Unexpected short inventory in the ETF itself.
            round_trips.append(
                {
                    "lot_id": len(round_trips) + 1,
                    "campaign_id": -1,
                    "symbol": sym,
                    "name": fill["name"],
                    "family": fill["family"],
                    "underlying": fill["underlying"],
                    "leverage": fill["leverage"],
                    "side_bias": fill["side_bias"],
                    "economic_side": fill["economic_side"],
                    "product": fill["product"],
                    "qty": remaining,
                    "entry_price": px,
                    "exit_price": np.nan,
                    "entry_time": ts,
                    "exit_time": pd.NaT,
                    "entry_fill_id": fill["fill_id"],
                    "exit_fill_id": np.nan,
                    "hold_minutes": np.nan,
                    "overnight": False,
                    "intraday": False,
                    "pnl": np.nan,
                    "return_pct": np.nan,
                    "entry_notional": remaining * px,
                    "exit_notional": np.nan,
                    "unmatched_short": True,
                }
            )

    open_lots = []
    for sym, dq in lots.items():
        for lot in dq:
            open_lots.append(
                {
                    "symbol": sym,
                    "qty": lot.qty_remaining,
                    "entry_price": lot.price,
                    "entry_time": lot.time,
                    "entry_fill_id": lot.fill_id,
                    "campaign_id": lot.campaign_id,
                    "entry_notional": lot.qty_remaining * lot.price,
                }
            )
        if campaigns[sym] is not None:
            close_campaign(sym)
            if campaign_rows:
                campaign_rows[-1]["status"] = "open"

    trips = pd.DataFrame(round_trips)
    camps = pd.DataFrame(campaign_rows)
    opens = pd.DataFrame(open_lots)
    if not trips.empty:
        trips = trips[~trips.get("unmatched_short", pd.Series(False, index=trips.index)).fillna(False)]
        trips = classify_outcomes(trips)
        trips = add_session_flags(trips)
        trips = add_reentry_flags(trips)
    if not camps.empty:
        camps = classify_outcomes(camps, prefix="")
        camps = add_session_flags(camps, entry_col="entry_time", exit_col="exit_time")
    return trips, camps, opens


def classify_outcomes(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    ret = out["return_pct"]
    pnl = out["pnl"]
    closed = ret.notna()
    out["winner"] = closed & (pnl > 0)
    out["loser"] = closed & (pnl <= 0)
    label = np.where(~closed, "open", "unclassified")
    label = np.where(closed & (ret >= LARGE_WIN_PCT), "large_winner", label)
    label = np.where(closed & (ret > 0) & (ret < LARGE_WIN_PCT), "small_winner", label)
    label = np.where(
        closed & (ret <= 0) & (ret > -SMALL_BAND_PCT), "small_loser", label
    )
    label = np.where(
        closed & (ret <= -SMALL_BAND_PCT) & (ret > CATASTROPHIC_LOSS_PCT),
        "large_loser",
        label,
    )
    label = np.where(closed & (ret <= CATASTROPHIC_LOSS_PCT), "catastrophic_loser", label)
    out["outcome"] = label
    return out


def add_session_flags(
    df: pd.DataFrame, entry_col: str = "entry_time", exit_col: str = "exit_time"
) -> pd.DataFrame:
    out = df.copy()

    def bucket(ts: pd.Timestamp) -> str:
        if pd.isna(ts):
            return "unknown"
        h, m = ts.hour, ts.minute
        minutes = h * 60 + m
        rth0 = 9 * 60 + 30
        rth1 = 16 * 60
        if minutes < rth0:
            return "premarket"
        if rth0 <= minutes < 10 * 60:
            return "open_30"
        if 10 * 60 <= minutes < 12 * 60:
            return "morning"
        if 12 * 60 <= minutes < 14 * 60:
            return "midday"
        if 14 * 60 <= minutes < rth1:
            return "afternoon"
        return "afterhours"

    out["entry_session"] = out[entry_col].map(bucket)
    out["exit_session"] = out[exit_col].map(bucket)
    hold = out["hold_minutes"]
    out["hold_bucket"] = pd.cut(
        hold,
        bins=[-0.01, 15, 60, 240, 1440, 4320, 1e9],
        labels=["<15m", "15-60m", "1-4h", "4-24h", "1-3d", ">3d"],
    )
    return out


def add_reentry_flags(trips: pd.DataFrame) -> pd.DataFrame:
    out = trips.sort_values(["entry_time", "lot_id"]).copy()
    out["minutes_since_prev_exit"] = np.nan
    out["prev_was_loss"] = False
    out["reentry_after_loss_30m"] = False
    out["reentry_after_loss_same_family"] = False
    closed = out[out["exit_time"].notna()].copy()
    for idx, row in out.iterrows():
        prior = closed[closed["exit_time"] <= row["entry_time"]]
        if prior.empty:
            continue
        sym_prior = prior[prior["symbol"] == row["symbol"]]
        if not sym_prior.empty:
            last_s = sym_prior.sort_values("exit_time").iloc[-1]
            dt_s = (row["entry_time"] - last_s["exit_time"]).total_seconds() / 60.0
            loss_s = bool(last_s["pnl"] <= 0)
            out.at[idx, "minutes_since_prev_exit"] = dt_s
            out.at[idx, "prev_was_loss"] = loss_s
            if loss_s and dt_s <= 30:
                out.at[idx, "reentry_after_loss_30m"] = True
        fam_prior = prior[prior["family"] == row["family"]]
        if fam_prior.empty:
            continue
        last_f = fam_prior.sort_values("exit_time").iloc[-1]
        dt_f = (row["entry_time"] - last_f["exit_time"]).total_seconds() / 60.0
        if bool(last_f["pnl"] <= 0) and dt_f <= 30:
            out.at[idx, "reentry_after_loss_same_family"] = True
    return out


def save_matched(trips: pd.DataFrame, camps: pd.DataFrame, opens: pd.DataFrame) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    trips.to_parquet(PROC_DIR / "round_trips.parquet", index=False)
    trips.to_csv(PROC_DIR / "round_trips.csv", index=False)
    camps.to_parquet(PROC_DIR / "campaigns.parquet", index=False)
    camps.to_csv(PROC_DIR / "campaigns.csv", index=False)
    opens.to_parquet(PROC_DIR / "open_lots.parquet", index=False)
    opens.to_csv(PROC_DIR / "open_lots.csv", index=False)
