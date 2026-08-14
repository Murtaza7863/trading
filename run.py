#!/usr/bin/env python3
"""Run the full historical-trade research pipeline. Not a live trading bot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import MARKET_DIR, OUT_DIR, PROC_DIR  # noqa: E402
from src.parse_orders import load_fills  # noqa: E402
from src.fifo import match_fifo, save_matched  # noqa: E402
from src.market_data import download_all  # noqa: E402
from src.features import annotate_trades  # noqa: E402
from src.analysis import run_analysis  # noqa: E402
from src.ml_models import walk_forward  # noqa: E402
from src.strategy import run_backtests  # noqa: E402
from src.report import make_figures, write_reports  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Intraday strategy research on historical fills")
    parser.add_argument("--skip-download", action="store_true", help="Use cached market parquet only")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("==> parsing fills")
    fills = load_fills()
    print(f"    filled rows: {len(fills)} symbols={sorted(fills['symbol'].unique())}")

    print("==> FIFO matching")
    trips, camps, opens = match_fifo(fills)
    save_matched(trips, camps, opens)
    print(f"    closed lots={trips['pnl'].notna().sum()} campaigns={len(camps)} open_lots={len(opens)}")
    print(f"    total realized P&L: {trips['pnl'].sum():.2f}")

    if not args.skip_download:
        print("==> downloading market data (Yahoo). empty downloads stay empty.")
        manifest = download_all(force=args.force_download)
        (MARKET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(
            f"    ok={len(manifest['ok'])} empty={len(manifest['empty'])} error={len(manifest['error'])}"
        )
    else:
        print("==> skipping download")

    print("==> features + MAE/MFE")
    featured = annotate_trades(trips)
    print(f"    feature interval counts:\n{featured['feat_interval'].value_counts(dropna=False)}")

    print("==> statistical research")
    tables = run_analysis(featured, camps)

    print("==> walk-forward ML")
    ml = walk_forward(featured)

    print("==> strategy backtests + baselines")
    backtest = run_backtests(featured)

    print("==> figures + reports")
    make_figures(featured, backtest)
    paths = write_reports(featured, camps, opens, tables, ml, backtest)
    print(f"    research: {paths['research']}")
    print(f"    spec:     {paths['spec']}")
    print("done. do not deploy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
