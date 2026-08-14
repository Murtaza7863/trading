"""Paths, instrument maps, costs, and research constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "Order Records_20260813231011.csv"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MARKET_DIR = DATA_DIR / "market"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "reports"
JOURNAL_PATH = DATA_DIR / "journal" / "trades.json"

TZ = "America/New_York"
RTH_START = (9, 30)
RTH_END = (16, 0)
PREMARKET_START = (4, 0)
AH_END = (20, 0)
OR_MINUTES = 30
FLAT_MINUTES_BEFORE_CLOSE = 5


def et_session(now=None) -> dict:
    """US equity session in America/New_York. Not a broker clock."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo(TZ)
    now = now or datetime.now(et)
    if now.tzinfo is None:
        now = now.replace(tzinfo=et)
    else:
        now = now.astimezone(et)
    mins = now.hour * 60 + now.minute
    pre = PREMARKET_START[0] * 60 + PREMARKET_START[1]
    rth = RTH_START[0] * 60 + RTH_START[1]
    close = RTH_END[0] * 60 + RTH_END[1]
    ah = AH_END[0] * 60 + AH_END[1]
    weekend = now.weekday() >= 5
    if weekend:
        kind = "weekend"
    elif pre <= mins < rth:
        kind = "premarket"
    elif rth <= mins < close:
        kind = "rth"
    elif close <= mins < ah:
        kind = "afterhours"
    else:
        kind = "overnight"
    return {
        "kind": kind,
        "et_local": now.strftime("%Y-%m-%dT%H:%M"),
        "et_label": now.strftime("%a %d %b %Y, %H:%M ET"),
        "after_close": kind in {"afterhours", "overnight", "weekend"},
        "morning_grind": kind == "rth" and 10 <= now.hour < 12,
        "premarket": kind == "premarket",
        "afterhours": kind == "afterhours",
        "weekend": weekend,
        "mins": mins,
    }


def expected_volume_frac(kind: str, mins: int, complete: bool = False) -> float:
    """Rough share of a full day's volume that has usually printed by now."""
    if complete or kind in {"overnight", "weekend"}:
        return 1.0
    pre = PREMARKET_START[0] * 60 + PREMARKET_START[1]
    rth = RTH_START[0] * 60 + RTH_START[1]
    close = RTH_END[0] * 60 + RTH_END[1]
    ah = AH_END[0] * 60 + AH_END[1]
    if kind == "premarket":
        return 0.02 + 0.08 * max(0, mins - pre) / max(1, rth - pre)
    if kind == "rth":
        return 0.10 + 0.82 * max(0, mins - rth) / max(1, close - rth)
    if kind == "afterhours":
        return 0.92 + 0.08 * max(0, mins - close) / max(1, ah - close)
    return 1.0


def expected_range_frac(kind: str, mins: int, complete: bool = False) -> float:
    """Share of a typical day's range that has usually printed by now."""
    if complete or kind in {"overnight", "weekend"}:
        return 1.0
    pre = PREMARKET_START[0] * 60 + PREMARKET_START[1]
    rth = RTH_START[0] * 60 + RTH_START[1]
    close = RTH_END[0] * 60 + RTH_END[1]
    ah = AH_END[0] * 60 + AH_END[1]
    if kind == "premarket":
        return 0.10 + 0.18 * max(0, mins - pre) / max(1, rth - pre)
    if kind == "rth":
        return 0.28 + 0.62 * max(0, mins - rth) / max(1, close - rth)
    if kind == "afterhours":
        return 0.90 + 0.10 * max(0, mins - close) / max(1, ah - close)
    return 1.0


# Fills present in the 2026-08-14 PDF export but missing from the CSV.
PDF_OVERLAY_FILLS = [
    {
        "symbol": "MUZ",
        "name": "Defiance Daily Target 2X Short MU ETF",
        "side": "SELL",
        "status": "Filled",
        "qty": 115.0,
        "price": 9.28,
        "limit_price": 9.28,
        "placed_time": "2026/08/13 11:22:26 EDT",
        "filled_time": "2026/08/13 11:22:38 EDT",
        "source": "pdf_overlay",
    },
    {
        "symbol": "MUU",
        "name": "Direxion Daily MU Bull 2X Shares",
        "side": "BUY",
        "status": "Filled",
        "qty": 33.0,
        "price": 32.6099,
        "limit_price": 32.67,
        "placed_time": "2026/08/13 11:23:45 EDT",
        "filled_time": "2026/08/13 11:23:46 EDT",
        "source": "pdf_overlay",
    },
]

# Traded product -> underlying used for signal generation.
INSTRUMENTS = {
    "MUU": {
        "underlying": "MU",
        "leverage": 2.0,
        "side_bias": "long",
        "family": "MU",
        "product": "levered_etf",
    },
    "MUZ": {
        "underlying": "MU",
        "leverage": -2.0,
        "side_bias": "short",
        "family": "MU",
        "product": "levered_etf",
    },
    "MULL": {
        "underlying": "MU",
        "leverage": 2.0,
        "side_bias": "long",
        "family": "MU",
        "product": "levered_etf",
    },
    "NVDL": {
        "underlying": "NVDA",
        "leverage": 2.0,
        "side_bias": "long",
        "family": "NVDA",
        "product": "levered_etf",
    },
    "NVDQ": {
        "underlying": "NVDA",
        "leverage": -2.0,
        "side_bias": "short",
        "family": "NVDA",
        "product": "levered_etf",
    },
    "SOXL": {
        "underlying": "SOXX",
        "leverage": 3.0,
        "side_bias": "long",
        "family": "SOXX",
        "product": "levered_etf",
    },
    "SOXS": {
        "underlying": "SOXX",
        "leverage": -3.0,
        "side_bias": "short",
        "family": "SOXX",
        "product": "levered_etf",
    },
    "SKUU": {
        "underlying": "000660.KS",
        "leverage": 2.0,
        "side_bias": "long",
        "family": "SKHYNIX",
        "product": "levered_etf",
        "sector_proxy": "SOXX",
        "note": "SK Hynix; Korean cash session, SOXX used as US-hours sector proxy",
    },
    "AVGU": {
        "underlying": "AVGO",
        "leverage": 2.0,
        "side_bias": "long",
        "family": "AVGO",
        "product": "levered_etf",
    },
    "AVL": {
        "underlying": "AVGO",
        "leverage": 2.0,
        "side_bias": "long",
        "family": "AVGO",
        "product": "levered_etf",
    },
    "AMDL": {
        "underlying": "AMD",
        "leverage": 2.0,
        "side_bias": "long",
        "family": "AMD",
        "product": "levered_etf",
    },
    "QCMU": {
        "underlying": "QCOM",
        "leverage": 2.0,
        "side_bias": "long",
        "family": "QCOM",
        "product": "levered_etf",
    },
    "CBRX": {
        "underlying": "CBRS",
        "leverage": 2.0,
        "side_bias": "long",
        "family": "CBRS",
        "product": "levered_etf",
        "note": "Cerebras Systems Inc.",
    },
    "CBRZ": {
        "underlying": "CBRS",
        "leverage": -2.0,
        "side_bias": "short",
        "family": "CBRS",
        "product": "levered_etf",
        "note": "Cerebras Systems Inc.",
    },
    "UVXY": {
        "underlying": "^VIX",
        "leverage": 1.5,
        "side_bias": "long",
        "family": "VIX",
        "product": "vol_etf",
        "note": "1.5x VIX short-term futures; path-dependent, not a clean 1.5x VIX",
    },
}

ETF_TICKERS = sorted(INSTRUMENTS.keys())
UNDERLYING_TICKERS = sorted(
    {v["underlying"] for v in INSTRUMENTS.values()}
    | {"SOXX", "SMH", "^IXIC", "SPY"}
)
SECTOR_TICKER = "SOXX"
MARKET_TICKER = "SPY"

# Yahoo Finance lookbacks (do not invent bars if a download fails).
YF_DAILY_START = "2025-01-01"
YF_HOURLY_START = "2025-10-01"
YF_END = "2026-08-15"

# Cost model used in strategy backtests (user fills already include real slippage).
COMMISSION_PER_SHARE = 0.0
SLIPPAGE_BPS_DEFAULT = 5.0
SLIPPAGE_BPS_GRID = [0.0, 5.0, 10.0, 20.0]

ASSUMED_EQUITY = 10_000.0
RISK_PCT_PER_TRADE = 0.01
MAX_SIMULTANEOUS = 2
MAX_PORTFOLIO_RISK_PCT = 0.02
COOLDOWN_MINUTES = 30
MAX_HOLD_MINUTES_V1 = 90

# Outcome labels on the traded ETF return.
LARGE_WIN_PCT = 5.0
CATASTROPHIC_LOSS_PCT = -10.0
SMALL_BAND_PCT = 5.0

RANDOM_SEED = 42
WALK_FORWARD_OOS_FRAC = 0.30
N_BOOTSTRAP = 2000

# Same-session volatility board. Rank where the tape is moving — not a buy list.
# Skip brand-new single-name 2x (CBRX, SKUU) and treat 3x SOXL/SOXS as last-resort.
WATCHLIST = [
    {"ticker": "MU", "name": "Micron", "group": "memory", "vehicle": "MUU", "vehicle_lev": 2, "ok_vehicle": True},
    {"ticker": "NVDA", "name": "NVIDIA", "group": "ai_gpu", "vehicle": "NVDL", "vehicle_lev": 2, "ok_vehicle": True},
    {"ticker": "AMD", "name": "AMD", "group": "ai_gpu", "vehicle": "AMDL", "vehicle_lev": 2, "ok_vehicle": True},
    {"ticker": "AVGO", "name": "Broadcom", "group": "ai_infra", "vehicle": "AVGU", "vehicle_lev": 2, "ok_vehicle": True},
    {"ticker": "QCOM", "name": "Qualcomm", "group": "semis", "vehicle": "QCMU", "vehicle_lev": 2, "ok_vehicle": True},
    {"ticker": "TSM", "name": "TSMC ADR", "group": "foundry", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "AMAT", "name": "Applied Materials", "group": "equipment", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "LRCX", "name": "Lam Research", "group": "equipment", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "KLAC", "name": "KLA", "group": "equipment", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "SNDK", "name": "Sandisk", "group": "memory", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "WDC", "name": "Western Digital", "group": "memory", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "SMCI", "name": "Super Micro", "group": "ai_infra", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "CRWV", "name": "CoreWeave", "group": "ai_infra", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "ARM", "name": "Arm", "group": "semis", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "INTC", "name": "Intel", "group": "semis", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "SOXX", "name": "iShares Semi ETF", "group": "sector", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "SMH", "name": "VanEck Semi ETF", "group": "sector", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
    {"ticker": "SPY", "name": "S&P 500", "group": "benchmark", "vehicle": None, "vehicle_lev": 1, "ok_vehicle": False},
]

WATCHLIST_SKIP = {
    "CBRX": "2x ETF listed 2026-05-28; overnighted at 6 days old, −$226",
    "CBRZ": "paired inverse of CBRX; same new-product ban",
    "CBRS": "IPO 2026-05-14; too young for this board",
    "SKUU": "2x ETF listed 2026-07-14; overnighted at 2 days old, −$97",
    "SOXL": "3x daily-reset; overnight cluster −$542 on the family",
    "SOXS": "−3x daily-reset; do not hold through prints or overnight",
    "UVXY": "VIX futures path, not a clean tape to hunt",
}
