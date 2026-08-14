"""Parse broker CSV fills and overlay the later PDF export."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import INSTRUMENTS, PDF_OVERLAY_FILLS, PROC_DIR, RAW_CSV, TZ

NY = ZoneInfo(TZ)


def _parse_symbol(raw: str) -> tuple[str, str]:
    lines = [ln.strip() for ln in str(raw).splitlines() if ln.strip()]
    symbol = lines[0] if lines else ""
    name = " ".join(lines[1:]) if len(lines) > 1 else ""
    return symbol, name


def _parse_qty(raw: str) -> tuple[float, float]:
    text = str(raw).replace(",", "").strip()
    if "/" not in text:
        q = float(text or 0)
        return q, q
    filled, total = text.split("/", 1)
    return float(filled), float(total)


def _parse_price(raw: str) -> tuple[float | None, float | None]:
    lines = [ln.strip() for ln in str(raw).splitlines() if ln.strip()]
    limit_price = None
    avg_price = None
    if lines:
        first = lines[0].replace("@", "").replace(",", "")
        if first not in {"", "-"}:
            limit_price = float(first)
        last = lines[-1].replace("@", "").replace(",", "")
        if last not in {"", "-"}:
            avg_price = float(last)
    return avg_price, limit_price


def _parse_ts(raw: str) -> pd.Timestamp | pd.NaT:
    text = str(raw).strip()
    if not text:
        return pd.NaT
    text = re.sub(r"\s+(EDT|EST|ET)$", "", text)
    try:
        dt = datetime.strptime(text, "%Y/%m/%d %H:%M:%S")
    except ValueError:
        try:
            dt = datetime.strptime(text, "%Y/%m/%d %H:%M")
        except ValueError:
            return pd.NaT
    return pd.Timestamp(dt.replace(tzinfo=NY))


def parse_csv(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_CSV
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for i, rec in enumerate(reader):
            symbol, name = _parse_symbol(rec.get("Symbol", ""))
            filled_qty, total_qty = _parse_qty(rec.get("Filled/Total Qty", "0/0"))
            avg_price, limit_price = _parse_price(rec.get("Price/Avg Price", ""))
            rows.append(
                {
                    "source": "csv",
                    "csv_row": i,
                    "symbol": symbol,
                    "name": name,
                    "side": str(rec.get("Side", "")).strip().upper(),
                    "status": str(rec.get("Status", "")).strip(),
                    "qty": filled_qty,
                    "total_qty": total_qty,
                    "price": avg_price,
                    "limit_price": limit_price,
                    "tif": str(rec.get("Time-in-Force", "")).strip(),
                    "placed_time": _parse_ts(rec.get("Placed Time", "")),
                    "filled_time": _parse_ts(rec.get("Filled Time", "")),
                }
            )
    return pd.DataFrame(rows)


def _overlay_row(item: dict) -> dict:
    return {
        "source": item["source"],
        "csv_row": -1,
        "symbol": item["symbol"],
        "name": item["name"],
        "side": item["side"],
        "status": item["status"],
        "qty": float(item["qty"]),
        "total_qty": float(item["qty"]),
        "price": float(item["price"]),
        "limit_price": item.get("limit_price"),
        "tif": "DAY",
        "placed_time": _parse_ts(item["placed_time"]),
        "filled_time": _parse_ts(item["filled_time"]),
    }


def _is_duplicate(a: pd.Series, b: dict) -> bool:
    if a["symbol"] != b["symbol"] or a["side"] != b["side"]:
        return False
    if abs(float(a["qty"]) - float(b["qty"])) > 1e-9:
        return False
    if pd.isna(a["filled_time"]) or pd.isna(b["filled_time"]):
        return False
    return abs((a["filled_time"] - b["filled_time"]).total_seconds()) <= 2


def merge_overlay(fills: pd.DataFrame) -> pd.DataFrame:
    extra = []
    for item in PDF_OVERLAY_FILLS:
        row = _overlay_row(item)
        if any(_is_duplicate(fills.iloc[i], row) for i in range(len(fills))):
            continue
        extra.append(row)
    if extra:
        fills = pd.concat([fills, pd.DataFrame(extra)], ignore_index=True)
    return fills


def annotate_instruments(fills: pd.DataFrame) -> pd.DataFrame:
    meta = fills["symbol"].map(INSTRUMENTS)
    unknown = fills.loc[meta.isna(), "symbol"].dropna().unique().tolist()
    if unknown:
        raise ValueError(f"Unmapped symbols (identify underlying before analysis): {unknown}")
    fills = fills.copy()
    fills["underlying"] = meta.map(lambda d: d["underlying"])
    fills["leverage"] = meta.map(lambda d: d["leverage"])
    fills["side_bias"] = meta.map(lambda d: d["side_bias"])
    fills["family"] = meta.map(lambda d: d["family"])
    fills["product"] = meta.map(lambda d: d["product"])
    fills["sector_proxy"] = meta.map(lambda d: d.get("sector_proxy"))
    fills["economic_side"] = fills["side_bias"].map({"long": 1, "short": -1})
    return fills


def load_fills() -> pd.DataFrame:
    fills = parse_csv()
    fills = merge_overlay(fills)
    fills = annotate_instruments(fills)
    fills["notional"] = fills["qty"] * fills["price"]
    filled = fills[fills["status"].str.lower() == "filled"].copy()
    filled = filled[filled["qty"] > 0]
    filled = filled.sort_values(["filled_time", "csv_row", "symbol"]).reset_index(drop=True)
    filled["fill_id"] = range(1, len(filled) + 1)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    fills.to_parquet(PROC_DIR / "all_orders.parquet", index=False)
    filled.to_parquet(PROC_DIR / "fills.parquet", index=False)
    filled.to_csv(PROC_DIR / "fills.csv", index=False)
    return filled
