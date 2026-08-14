#!/usr/bin/env python3
"""Build a same-session volatility board. Not a buy list and not a live bot."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.watchlist import build_watchlist, write_watchlist  # noqa: E402

TAPE_NOTE = (
    "As of 13–14 Aug 2026 the public tape in this group was memory: "
    "Sandisk / Western Digital / Micron / SK Hynix ripped on NAND/HBM commentary "
    "(Sandisk investor day, KeyBanc tightness comments earlier in the week). "
    "That is *where* the range is. It is not a reason to overnight a 2x short or a new ETF."
)


def main() -> int:
    print("==> scoring watchlist names from Yahoo")
    df = build_watchlist()
    paths = write_watchlist(df, tape_note=TAPE_NOTE)
    print(df[["ticker", "fuel_score", "lean", "setup", "pm_ret_pct", "atr_used", "spent", "fuel_note"]].to_string(index=False))
    print(f"    csv: {paths['csv']}")
    print(f"    md:  {paths['md']}")
    print("done. look, don't buy from the rank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
