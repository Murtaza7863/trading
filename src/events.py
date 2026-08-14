"""Public calendars and product-age tags. Known at or before the fill — not lookahead.

Sources are cited in EVENT_SOURCES. Context-only tags (HBM squeeze week) are
narrative, not entry filters.
"""

from __future__ import annotations

import pandas as pd

from .config import TZ

# After-close / local-print timestamps in America/New_York.
EARNINGS = [
    {
        "name": "NVDA_FQ1_FY27",
        "ts": "2026-05-20 16:00",
        "families": ("NVDA",),
        "note": "Record $81.6B revenue; typically after the cash close.",
    },
    {
        "name": "MU_FQ3_FY26",
        "ts": "2026-06-24 16:01",
        "families": ("MU", "SOXX", "SKHYNIX"),
        "note": "Record $41.46B revenue, ~$50B FQ4 guide; +15% after hours.",
    },
    {
        "name": "SKHYNIX_2Q26",
        "ts": "2026-07-28 19:00",
        "families": ("SKHYNIX", "MU", "SOXX"),
        "note": "Record quarter that missed consensus; KR shares −9.6% on Jul 29.",
    },
]

FOMC_DECISIONS = [
    "2026-04-29 14:00",
    "2026-06-17 14:00",
    "2026-07-29 14:00",
]

# Single-name 2x products that listed during the book.
PRODUCT_INCEPTION = {
    "CBRX": "2026-05-28",
    "CBRZ": "2026-05-28",
    "SKUU": "2026-07-14",
}

# News-week overlap for narrative only. Not a tradable filter.
HBM_SQUEEZE = ("2026-08-10 00:00", "2026-08-13 16:00")
HBM_FAMILIES = ("MU", "SKHYNIX", "SOXX")

NEW_PRODUCT_DAYS = 14
EARNINGS_HOLD_HOURS = 36
FOMC_HOLD_HOURS = 18
FADE_MEMORY_DAYS = 21

EVENT_SOURCES = [
    {
        "event": "NVDA_FQ1_FY27",
        "date": "2026-05-20",
        "url": "https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Announces-Financial-Results-for-First-Quarter-Fiscal-2027/",
        "note": "NVIDIA $81.6B Q1 FY27 revenue.",
    },
    {
        "event": "CBRS_IPO",
        "date": "2026-05-14",
        "url": "https://www.cnbc.com/2026/05/14/cerebras-cbrs-stock-trade-nasdaq-ipo.html",
        "note": "Cerebras IPO $185, opened ~$350.",
    },
    {
        "event": "CBRX_INCEPTION",
        "date": "2026-05-28",
        "url": "https://www.tradretfs.com/news-and-media/tradr-launches-two-leveraged-etfs-on-cerebras-systems",
        "note": "Tradr 2x long CBRS (CBRX) listed ~2 weeks after the IPO.",
    },
    {
        "event": "FOMC_2026-06-17",
        "date": "2026-06-17",
        "url": "https://www.federalreserve.gov/monetarypolicy/files/monetary20260617a1.pdf",
        "note": "Fed held 3.50–3.75%, unanimous.",
    },
    {
        "event": "MU_FQ3_FY26",
        "date": "2026-06-24",
        "url": "https://www.globenewswire.com/news-release/2026/06/24/3317151/14450/en/Micron-Technology-Inc-Reports-Record-Results-for-the-Third-Quarter-of-Fiscal-2026.html",
        "note": "Micron FQ3 FY26 $41.46B rev; stock +15% AH (CNBC).",
    },
    {
        "event": "MU_FQ3_FY26_AH",
        "date": "2026-06-24",
        "url": "https://www.cnbc.com/2026/06/24/micron-mu-earnings-report-q3-2026.html",
        "note": "CNBC: beat and $50B guide; +15% extended.",
    },
    {
        "event": "SKUU_INCEPTION",
        "date": "2026-07-14",
        "url": "https://graniteshares.com/press/sk-hynix-etfs-skuu-and-skdd/",
        "note": "GraniteShares 2x SK Hynix (SKUU) launched Jul 14, 4 days after Nasdaq ADR.",
    },
    {
        "event": "SOXS_REVERSE_SPLIT",
        "date": "2026-07-15",
        "url": "https://www.direxion.com/press-release/direxion-to-split-nine-etfs",
        "note": "SOXS 1-for-10 reverse split; daily-reset −3x decay made visible.",
    },
    {
        "event": "SKHYNIX_2Q26",
        "date": "2026-07-29",
        "url": "https://news.skhynix.com/en/q2-2026-business-results/",
        "note": "Record ₩79.3T revenue that still missed; shares −9.6%.",
    },
    {
        "event": "FOMC_2026-07-29",
        "date": "2026-07-29",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260729a.htm",
        "note": "Fed held 3.50–3.75%, 9–3 vote.",
    },
    {
        "event": "MU_HBM_SQUEEZE",
        "date": "2026-08-12",
        "url": "https://invezz.com/news/2026/08/12/why-are-micron-sk-hynix-stocks-rallying-today-after-coreweave-super-micro-earnings/",
        "note": "KeyBanc: 2027 memory tighter than 2026; CoreWeave/SMCI AI-infra beat.",
    },
]


def _ts(s: str) -> pd.Timestamp:
    t = pd.Timestamp(s)
    if t.tzinfo is None:
        t = t.tz_localize(TZ)
    else:
        t = t.tz_convert(TZ)
    return t


def event_calendar() -> pd.DataFrame:
    rows = []
    for ev in EARNINGS:
        rows.append(
            {
                "kind": "earnings",
                "name": ev["name"],
                "ts": ev["ts"],
                "families": ",".join(ev["families"]),
                "note": ev["note"],
            }
        )
    for s in FOMC_DECISIONS:
        rows.append({"kind": "fomc", "name": "FOMC", "ts": s, "families": "*", "note": "Scheduled 14:00 ET decision."})
    for ticker, start in PRODUCT_INCEPTION.items():
        rows.append(
            {
                "kind": "product_inception",
                "name": ticker,
                "ts": start + " 09:30",
                "families": ticker,
                "note": f"New single-name 2x; {NEW_PRODUCT_DAYS}d overnight ban.",
            }
        )
    rows.append(
        {
            "kind": "context",
            "name": "MU_HBM_SQUEEZE",
            "ts": HBM_SQUEEZE[0],
            "families": ",".join(HBM_FAMILIES),
            "note": "Context only — not used as an entry filter.",
        }
    )
    return pd.DataFrame(rows)


def sources_table() -> pd.DataFrame:
    return pd.DataFrame(EVENT_SOURCES)


def _as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    mapped = s.map(
        lambda x: str(x).strip().lower() in {"1", "true", "yes", "t"}
        if pd.notna(x)
        else False
    )
    return mapped.astype(bool)


def tag_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    entry = pd.to_datetime(out["entry_time"], utc=True).dt.tz_convert(TZ)
    exit_t = pd.to_datetime(out["exit_time"], utc=True).dt.tz_convert(TZ)
    fam = out["family"].astype(str)
    sym = out["symbol"].astype(str)
    overnight = _as_bool(out["overnight"])
    econ = pd.to_numeric(out.get("economic_side"), errors="coerce")

    earn_flag = pd.Series(False, index=out.index)
    earn_name = pd.Series("", index=out.index)
    for ev in EARNINGS:
        ts = _ts(ev["ts"])
        lo, hi = ts, ts + pd.Timedelta(hours=EARNINGS_HOLD_HOURS)
        hit = fam.isin(ev["families"]) & (entry <= hi) & (exit_t >= lo)
        earn_flag |= hit
        earn_name = earn_name.mask(hit & (earn_name != ""), earn_name + ";" + ev["name"])
        earn_name = earn_name.mask(hit & (earn_name == ""), ev["name"])

    fomc_flag = pd.Series(False, index=out.index)
    for s in FOMC_DECISIONS:
        ts = _ts(s)
        lo, hi = ts - pd.Timedelta(hours=2), ts + pd.Timedelta(hours=FOMC_HOLD_HOURS)
        fomc_flag |= (entry <= hi) & (exit_t >= lo)

    new_prod = pd.Series(False, index=out.index)
    age_days = pd.Series(pd.NA, index=out.index, dtype="Float64")
    for ticker, start in PRODUCT_INCEPTION.items():
        born = _ts(start + " 09:30")
        m = sym == ticker
        age_days.loc[m] = (entry[m] - born).dt.total_seconds() / 86400.0
        new_prod |= m & (age_days < NEW_PRODUCT_DAYS)

    mu_earn = _ts("2026-06-24 16:01")
    fade_memory = (
        overnight
        & (econ == -1)
        & fam.isin(["MU", "SOXX", "SKHYNIX"])
        & (entry >= mu_earn)
        & (entry <= mu_earn + pd.Timedelta(days=FADE_MEMORY_DAYS))
    )

    hbm_lo, hbm_hi = _ts(HBM_SQUEEZE[0]), _ts(HBM_SQUEEZE[1])
    hbm = fam.isin(HBM_FAMILIES) & (entry <= hbm_hi) & (exit_t >= hbm_lo)

    out["event_earnings"] = earn_flag
    out["event_earnings_name"] = earn_name
    out["event_fomc"] = fomc_flag
    out["event_new_product"] = new_prod
    out["product_age_days"] = age_days
    out["event_fade_memory_post_mu_earn"] = fade_memory
    out["event_hbm_squeeze"] = hbm
    out["event_any"] = earn_flag | fomc_flag | new_prod | fade_memory
    out["event_overnight"] = out["event_any"] & overnight
    return out
