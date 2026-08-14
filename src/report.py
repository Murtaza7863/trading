"""Figures, research markdown, and the surviving-rules spec."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ASSUMED_EQUITY, FIG_DIR, PROC_DIR, REPORT_DIR, TABLE_DIR


def _fmt(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(trades: pd.DataFrame, backtest: dict) -> list[Path]:
    paths = []
    closed = trades[trades["pnl"].notna()].copy()
    if closed.empty:
        return paths

    fig, ax = plt.subplots(figsize=(9, 4))
    eq = ASSUMED_EQUITY + closed.sort_values("exit_time").set_index("exit_time")["pnl"].cumsum()
    ax.plot(eq.index, eq.values, color="#1f4e79", lw=1.6)
    ax.axhline(ASSUMED_EQUITY, color="#888", lw=0.8)
    ax.set_title("User realized equity (FIFO lots, $10k starting mark)")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Exit time")
    paths.append(_save(fig, "user_equity.png"))

    fig, ax = plt.subplots(figsize=(9, 3.6))
    peak = eq.cummax()
    dd = eq / peak - 1
    ax.plot(dd.index, dd.values * 100, color="#8b1e1e", lw=1.4)
    ax.set_title("User drawdown from peak realized equity")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Exit time")
    paths.append(_save(fig, "user_drawdown.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    fam = closed.groupby("family")["pnl"].sum().sort_values()
    ax.barh(fam.index.astype(str), fam.values, color=["#8b1e1e" if v < 0 else "#1f7a4c" for v in fam.values])
    ax.set_title("Total P&L by underlying family")
    ax.set_xlabel("P&L ($)")
    paths.append(_save(fig, "pnl_by_family.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    hb = closed.groupby("hold_bucket", observed=False)["pnl"].sum()
    ax.bar([str(x) for x in hb.index], hb.values, color="#1f4e79")
    ax.set_title("Total P&L by holding-period bucket")
    ax.set_ylabel("P&L ($)")
    ax.tick_params(axis="x", rotation=30)
    paths.append(_save(fig, "pnl_by_hold.png"))

    if "mae_pct" in closed.columns and closed["mae_pct"].notna().any():
        fig, ax = plt.subplots(figsize=(6.5, 6))
        c = np.where(closed["pnl"] > 0, "#1f7a4c", "#8b1e1e")
        ax.scatter(closed["mae_pct"], closed["mfe_pct"], c=c, alpha=0.7, s=28)
        ax.axhline(0, color="#888", lw=0.6)
        ax.axvline(0, color="#888", lw=0.6)
        ax.set_xlabel("MAE (%)")
        ax.set_ylabel("MFE (%)")
        ax.set_title("MAE vs MFE on traded ETF (green=winner)")
        paths.append(_save(fig, "mae_mfe.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(closed.loc[closed["pnl"] > 0, "return_pct"].dropna(), bins=20, alpha=0.7, label="winners", color="#1f7a4c")
    ax.hist(closed.loc[closed["pnl"] <= 0, "return_pct"].dropna(), bins=20, alpha=0.7, label="losers", color="#8b1e1e")
    ax.set_title("Winner vs loser return distributions")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Count")
    ax.legend()
    paths.append(_save(fig, "return_distributions.png"))

    seq = closed.sort_values("exit_time")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(seq)), seq["pnl"].cumsum().values, color="#1f4e79")
    ax.set_title("User trade-by-trade cumulative P&L")
    ax.set_xlabel("Closed lot # (chronological by exit)")
    ax.set_ylabel("Cumulative P&L ($)")
    paths.append(_save(fig, "user_cum_pnl.png"))

    seq_eq = backtest.get("strategy_equity")
    if isinstance(seq_eq, pd.Series) and len(seq_eq):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(seq_eq.index, seq_eq.values, color="#1f4e79", lw=1.6)
        ax.axhline(ASSUMED_EQUITY, color="#888", lw=0.8)
        ax.set_title("Systematic strategy equity (research backtest)")
        ax.set_ylabel("Equity ($)")
        ax.set_xlabel("Date")
        paths.append(_save(fig, "strategy_equity.png"))
        peak = seq_eq.cummax()
        dd = seq_eq / peak - 1
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(dd.index, dd.values * 100, color="#8b1e1e")
        ax.set_title("Systematic strategy drawdown")
        ax.set_ylabel("Drawdown (%)")
        paths.append(_save(fig, "strategy_drawdown.png"))

    return paths


def _md_table(df: pd.DataFrame, cols=None, n=12) -> str:
    if df is None or df.empty:
        return "_no data_\n"
    use = df.copy()
    if cols:
        cols = [c for c in cols if c in use.columns]
        use = use[cols]
    use = use.head(n)
    fmt = use.copy()
    for c in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[c]):
            fmt[c] = fmt[c].map(lambda x: _fmt(x, 3))
    return "|" + "|".join(map(str, fmt.columns)) + "|\n|" + "|".join(["---"] * len(fmt.columns)) + "|\n" + "\n".join(
        "|" + "|".join(map(str, row)) + "|" for row in fmt.itertuples(index=False)
    ) + "\n"


def write_reports(trades: pd.DataFrame, camps: pd.DataFrame, opens: pd.DataFrame, tables: dict, ml: dict, backtest: dict) -> dict[str, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    closed = trades[trades["pnl"].notna()].copy()
    summary = tables.get("overall")
    hyp = tables.get("hypotheses", pd.DataFrame())
    by_fam = tables.get("by_family", pd.DataFrame())
    by_sym = tables.get("by_symbol", pd.DataFrame())
    overlays = tables.get("rule_overlays", pd.DataFrame())
    stops = tables.get("stop_grid", pd.DataFrame())
    compare = backtest.get("compare", pd.DataFrame())
    ml_met = ml.get("metrics", pd.DataFrame()) if ml else pd.DataFrame()

    oos_ok = False
    candidate = False
    oos_note = "The systematic strategy did not demonstrate a robust out-of-sample edge versus simple baselines."
    if isinstance(compare, pd.DataFrame) and not compare.empty:
        row = compare.set_index("name")
        try:
            strat = row.loc["strategy_oos"]
            ema = None
            for key in ("ema_crossover_oos", "ema_crossover"):
                if key in row.index:
                    ema = row.loc[key]
                    break
            rnd = None
            for key in ("random_entry_oos", "random_entry"):
                if key in row.index:
                    rnd = row.loc[key]
                    break
            rnd_x = row.loc["random_exit_oos"] if "random_exit_oos" in row.index else None
            beats_random = rnd is not None and strat["total_pnl"] > rnd["total_pnl"] and strat["n"] > 5
            beats_exit = rnd_x is None or strat["total_pnl"] > rnd_x["total_pnl"]
            beats_ema = ema is None or strat["total_pnl"] > ema["total_pnl"]
            positive = strat["total_pnl"] > 0
            slip_path = TABLE_DIR / "strategy_slippage_sensitivity.csv"
            survives_20bps = False
            if slip_path.exists():
                slip = pd.read_csv(slip_path)
                s20 = slip.loc[slip["slippage_bps"] == 20.0, "total_pnl"]
                survives_20bps = bool(len(s20) and float(s20.iloc[0]) > 0)
            candidate = bool(beats_random and beats_ema and beats_exit and positive)
            oos_ok = bool(candidate and survives_20bps)
        except Exception:
            oos_ok = False
            candidate = False
        if oos_ok:
            oos_note = "Out-of-sample the engine beat matched EMA/random baselines and survived 20 bps costs. Sample is still one window; do not deploy without a second OOS period."
        elif candidate:
            oos_note = (
                "A candidate engine beat OOS EMA and random entries in this single window, "
                "but it is **not robust**: 20 bps slippage flips OOS P&L negative, "
                "parameters were selected from an 18-cell in-sample grid, and there is only one OOS slice. "
                "No deployable edge is claimed."
            )
        else:
            oos_note = "Out-of-sample, the systematic engine did **not** clear the usefulness bar versus matched baselines. No deployable edge was found."

    ml_skill = False
    if isinstance(ml_met, pd.DataFrame) and not ml_met.empty:
        hold = ml_met[(ml_met["split"] == "holdout_oos") & (ml_met["n"] >= 40)]
        if not hold.empty and hold["auc"].max() > 0.60:
            ml_skill = True

    research = []
    research.append("# Trade History Reverse-Engineering Report\n")
    research.append("**Not live trading. Not investment advice.** The goal is to test whether a repeatable statistical edge exists in this book, not to prove profitability.\n")
    research.append("## Data lineage\n")
    research.append(f"- Authoritative CSV: `Order Records_20260813231011.csv`")
    research.append("- Overlay from 2026-08-14 PDF: MUZ SELL 115 @ 9.28 (2026-08-13 11:22:38 EDT); MUU BUY 33 @ 32.6099 (open lot).")
    research.append("- Matching: FIFO per symbol. Inverse ETFs are long the ETF shares; `economic_side` maps bull/bear products to underlying direction.")
    research.append("- Market data: Yahoo Finance via yfinance. 1-minute (7d), 5/15-minute (60d), 60-minute, daily. Missing bars are left missing; nothing is interpolated.\n")
    research.append(f"- Closed FIFO lots: **{len(closed)}**. Open lots: **{len(opens)}**.")
    if not opens.empty:
        research.append("Open inventory at export:\n")
        research.append(_md_table(opens, ["symbol", "qty", "entry_price", "entry_time", "entry_notional"]))
    research.append("\n## Headline book statistics\n")
    research.append(_md_table(summary))
    research.append(f"- Catastrophic losers (≤ -10%): see `by_outcome`.")
    research.append("\n## Expectancy by instrument / family\n")
    research.append(_md_table(by_fam, ["group", "n", "win_rate", "total_pnl", "expectancy_pct", "profit_factor"]))
    research.append(_md_table(by_sym, ["group", "n", "win_rate", "total_pnl", "expectancy_pct", "profit_factor"], n=20))
    research.append("\n## Intraday vs overnight, hold, session, long vs short\n")
    for key in ["by_overnight", "by_hold_bucket", "by_entry_session", "by_economic_side"]:
        research.append(f"### {key}\n")
        research.append(_md_table(tables.get(key, pd.DataFrame())))
    research.append("\n## Hypothesis tests (Mann-Whitney on returns)\n")
    research.append("p-values are unadjusted. Treat as exploratory.\n")
    research.append(_md_table(hyp, ["hypothesis", "n_a", "n_b", "mean_a", "mean_b", "p_value", "significant_0_05"], n=20))
    research.append("\n## Features that differ between winners and losers\n")
    research.append(_md_table(tables.get("winner_loser_features", pd.DataFrame()), n=20))
    research.append("\n## RSI / EMA / VWAP / relative volume (features, not rules)\n")
    research.append(_md_table(tables.get("rsi_buckets", pd.DataFrame())))
    research.append(_md_table(tables.get("ema_vwap_rvol", pd.DataFrame())))
    research.append("\n## Counterfactual stops and profit-taking on the same entries\n")
    research.append("A stop at S% is assumed hit if MAE ≤ -S during the hold. If MAE and MFE share the same bar, order is unknown and the actual exit is kept.\n")
    research.append(_md_table(stops))
    research.append(_md_table(tables.get("takeprofit_no_stop", pd.DataFrame())))
    research.append("\n## Holding-period decay\n")
    research.append("This splits *realized* exits by hold length. It is not a mark-to-market flatten at the cap.\n")
    research.append(_md_table(tables.get("hold_decay", pd.DataFrame()), n=15))
    research.append("\n## Public tape vs this book\n")
    research.append(
        "Lots were tagged against **public calendars and product inception dates** "
        "(earnings 8-Ks, FOMC statements, Tradr/GraniteShares launch notices). "
        "The August HBM-tightness week is labeled as **context only** — it is not a filter "
        "you could have coded from a calendar in advance.\n"
    )
    research.append("These tags explain *why overnight 2x/3x died*. They are not an entry recipe.\n")
    research.append("### Event attribution\n")
    research.append(
        "Read this table as **damage diagnosis**, not as a filter to promote. "
        "Calendar-tagged lots were **net +$124**. Overnight lots *without* a dated event were **−$1,131**. "
        "The only calendar slice that is unambiguously toxic is **brand-new single-name 2x** (CBRX 6 days after listing, SKUU 2 days, combined −$322, 0/2). "
        "Holding SOXS through Micron’s 16:01 ET print (−$81) is the earnings smoking gun, but the rest of the earnings window (NVDA 5/20, MU same-session 6/25, SK Hynix 7/29) was green. "
        "Fading MU/SOXX overnight for 21 days after the 6/24 8-K was **+$380** — lucky inventory, not a setup. "
        "The August MUZ short overlapping KeyBanc/CoreWeave/SMCI HBM comments (−$492) is **context**, not a pre-coded calendar.\n"
    )
    research.append(
        _md_table(
            tables.get("event_attribution", pd.DataFrame()),
            ["group", "n", "win_rate", "total_pnl", "expectancy_pct", "profit_factor", "share_of_book_pnl"],
            n=15,
        )
    )
    research.append("### Tagged lots (calendar event or HBM-squeeze week)\n")
    research.append(
        _md_table(
            tables.get("event_lots", pd.DataFrame()),
            [
                "symbol",
                "family",
                "entry_time",
                "exit_time",
                "overnight",
                "pnl",
                "return_pct",
                "event_earnings_name",
                "event_new_product",
                "event_fade_memory_post_mu_earn",
                "event_hbm_squeeze",
            ],
            n=25,
        )
    )
    research.append("### Sources\n")
    src = tables.get("event_sources", pd.DataFrame())
    if isinstance(src, pd.DataFrame) and not src.empty:
        for _, r in src.iterrows():
            research.append(f"- {r['date']} **{r['event']}**: {r['note']} — {r['url']}")
        research.append("")
    research.append("\n## Behavioral overlays on the user's entries\n")
    research.append("These are filters on trades the user already took. They are **not** a standalone signal generator.\n")
    research.append(_md_table(overlays, ["group", "n", "win_rate", "total_pnl", "expectancy_pct", "profit_factor"], n=25))
    research.append("\n## Machine learning (chronological walk-forward)\n")
    research.append("Features are information available at entry. Observations are **not** shuffled. Sample size is small; AUC near 0.5 means no skill.\n")
    if isinstance(ml_met, pd.DataFrame) and not ml_met.empty:
        research.append(_md_table(ml_met, ["model", "split", "fold", "n", "base_rate", "auc", "ap", "brier"], n=30))
    research.append(f"- Detectable OOS classification skill (max holdout AUC > 0.60): **{'yes' if ml_skill else 'no'}**.")
    imp = ml.get("importance") if ml else None
    if isinstance(imp, pd.DataFrame) and not imp.empty:
        research.append("\n### Feature importance (holdout models)\n")
        sub = imp[imp["fold"].astype(str) == "chrono_70_30"].head(15)
        research.append(_md_table(sub, ["model", "feature", "importance", "kind"]))
    research.append("\n## Systematic strategy vs baselines\n")
    research.append(f"- Assumed research equity: ${ASSUMED_EQUITY:,.0f}. Costs: {backtest.get('wf', {}).get('best_params', {})}.")
    research.append(_md_table(compare, n=15))
    research.append("### Buy-and-hold (full sample)\n")
    research.append(_md_table(pd.DataFrame(backtest.get("buyhold", [])), n=15))
    research.append("### Buy-and-hold (OOS window)\n")
    research.append(_md_table(pd.DataFrame(backtest.get("buyhold_oos", [])), n=15))
    research.append("\n## Refinement: your book, early vs late lots\n")
    research.append(
        "Filters that can be applied to *your* entries, split at the same date as the engine. "
        "If a filter only works on early lots, it is not a pattern you can keep using.\n"
    )
    chrono = tables.get("user_filters_chrono", pd.DataFrame())
    research.append(_md_table(chrono, ["group", "split", "n", "win_rate", "total_pnl", "expectancy_pct", "profit_factor"], n=50))
    research.append("\n## Refinement: systematic variants (frozen OOS cut)\n")
    research.append(
        "Trend vs fade (mean-revert to VWAP), with/without stops, with/without SOXX. "
        "A variant is only interesting if OOS P&L is positive *and* it beats a random exit of its own entries.\n"
    )
    variants = backtest.get("variants", pd.DataFrame())
    if isinstance(variants, pd.DataFrame) and not variants.empty:
        research.append(
            _md_table(
                variants,
                ["variant", "split", "n", "total_pnl", "win_rate", "profit_factor", "random_exit_pnl", "beats_own_random_exit"],
                n=20,
            )
        )
    research.append(f"\n**Usefulness verdict:** {oos_note}\n")
    research.append("\n## Data limitations\n")
    research.append("- Yahoo 5-minute history is typically ~60 calendar days; older lots fall back to 60-minute or daily bars. MAE/MFE is coarser on those trades.")
    research.append("- Levered-ETF vs underlying tracking error, borrow, and overnight decay are not fully modeled. Version-1 engine forbids overnight.")
    research.append("- SKUU uses 000660.KS when available, otherwise SOXX as a US-hours sector proxy.")
    research.append("- UVXY is not a clean linear VIX product.")
    research.append("- Yahoo OHLC is often split-adjusted while broker fills are not; MAE/MFE rescales the path to the fill when the ratio is far from 1. Path windows that start too late after entry are skipped rather than truncated.")
    research.append("- One MUU 0.5-share fill near $1,100 is treated as a real fill (likely pre-split); prices are not adjusted.")
    research.append("- Multiple-testing: many slices were examined. Isolated p < 0.05 results are not a license to trade.\n")
    research.append("## Figures\n")
    research.append("Saved under `outputs/figures/`.\n")
    research_path = REPORT_DIR / "RESEARCH_REPORT.md"
    research_path.write_text("\n".join(research))

    spec = []
    spec.append("# Trading System Specification (surviving rules only)\n")
    spec.append("**Status: RESEARCH / DO NOT DEPLOY.**\n")
    spec.append(oos_note + "\n")
    spec.append("## What is allowed to appear here\n")
    spec.append("Only rules that either (a) improved the user's historical book with a clear economic rationale and were not contradicted out of sample, or (b) were selected on an in-sample grid and then evaluated out of sample on the systematic engine. Indicators are not hard-coded unless they survived that process.\n")

    spec.append("## Hard constraints (version 1, all variants)\n")
    spec.append("These are risk engineering, not alpha. They exist to remove the failure modes the book actually exhibited (oversized losses, overnight leveraged decay, averaging down, immediate re-entry).\n")
    spec.append("1. Trade only from the **underlying** (MU, NVDA, SOXX). Execute conceptually via the 2x/3x products; size from stop distance.")
    spec.append("2. **No overnight** inventory in leveraged products.")
    spec.append("3. **No averaging down.** One entry, one invalidation.")
    spec.append("4. Flatten **5 minutes before the cash close**.")
    spec.append("5. After a stop-out, **cooldown** in that name. 30 minutes is the conservative discretionary constraint; the in-sample grid selected 0 minutes, so it is not authorized as alpha.")
    spec.append("6. Risk **1% of equity** per trade; **max 2** positions; **max 2%** simultaneous portfolio risk.")
    spec.append("7. Include **≥5 bps slippage per side** in any forward test. If 10–20 bps kill the edge, there is no edge.")
    spec.append("8. Do not use RSI or EMA as standalone entry rules unless the feature tables show a stable bucket and the OOS engine still works with them **removed**.")
    spec.append("9. **Do not overnight a single-name 2x ETF in its first 14 sessions** (CBRX was 6 days old; SKUU was 2). That slice is 0/2 and −$322.")
    spec.append("10. **Do not hold inverse 3x through a scheduled earnings print** of the underlying or a close proxy (SOXS into MU 2026-06-24 16:01 ET). Flatten before the 8-K, do not invent a 21-day post-print fade ban — those later lots were net green and are not a rule.")
    spec.append("11. FOMC day: flattening leveraged inventory before 14:00 ET is risk hygiene. n=3 FOMC-overlapping lots were green here; do not treat that as an edge.")
    spec.append("12. Headlines that memory stays tight (KeyBanc / CoreWeave / SMCI, Aug 10–13 2026) are a reason **not** to sit short MU overnight. They are not a long-MU signal generator.\n")

    # Pull overlay improvements
    keep_filters = []
    if isinstance(overlays, pd.DataFrame) and not overlays.empty and "group" in overlays.columns:
        base = overlays[overlays["group"] == "baseline_all_lots"]
        base_pnl = float(base["total_pnl"].iloc[0]) if len(base) else np.nan
        for _, r in overlays.iterrows():
            if r["group"] == "baseline_all_lots":
                continue
            if r["n"] < 15:
                continue
            if r["group"] in {
                "hold_lt_60m",
                "hold_lt_240m",
                "mu_nvda_only",
                "no_earnings_window",
                "no_earnings_overnight",
                "no_new_2x_product",
                "no_fade_memory_post_mu_earn",
                "no_fomc_overnight",
                "no_event_overnight",
            }:
                # hold filters are not knowable at entry; MU/NVDA failed the later window;
                # calendar overlays are listed separately as risk constraints, not alpha.
                continue
            if (
                r["total_pnl"] > 0
                and np.isfinite(base_pnl)
                and r["total_pnl"] > base_pnl
                and r["expectancy_pct"] > float(base["expectancy_pct"].iloc[0])
            ):
                keep_filters.append(r["group"])
    spec.append("## Discretionary-book overlays (only if you keep taking similar trades)\n")
    if keep_filters:
        spec.append("On this sample, the following filters raised both total P&L and expectancy versus trading every historical lot:\n")
        for g in keep_filters:
            spec.append(f"- `{g}`")
        spec.append("\nThese are **not** validated as a signal generator. They only say: if the same entries appear, dropping the complementary set helped historically.\n")
    else:
        spec.append("No overlay simultaneously improved total P&L and expectancy with n≥15 versus the full book. Do not add discretionary filters that were not robust here.\n")

    spec.append("## Systematic engine\n")
    bp = backtest.get("wf", {}).get("best_params", {})
    spec.append(f"In-sample selected parameters: `{bp}`.\n")
    if oos_ok:
        spec.append("### Regime\n")
        spec.append("- Bullish trend: EMA9>EMA20>EMA50, price above session VWAP, not in volatility contraction.")
        spec.append("- Bearish trend: inverse.")
        spec.append("- Else: **no trade**.\n")
        spec.append("### Setup\n")
        spec.append("- Long: bullish regime, close > VWAP, not more than 1.5 ATR extended from VWAP, relative volume not starved, momentum or VWAP/EMA20 reclaim, no opening-range/breakout failure.")
        spec.append("- Short: inverse.")
        spec.append("- Signals from underlying only.\n")
        spec.append("### Exit\n")
        spec.append(f"- Stop: {bp.get('stop_atr', 1.5)} ATR from entry on the underlying (P&L scaled by product leverage).")
        spec.append("- Target: 2R, else time stop, else EOD flatten.")
        spec.append(f"- Max hold: {bp.get('max_hold_min', 90)} minutes.\n")
    else:
        spec.append("**No systematic entry/exit rule set is authorized.** Trend-following with stops lost to random exits of the same entries. Dropping SOXX, fading VWAP, and removing stops were tested on a frozen OOS cut; none is a repeatable edge. No-stop trend looked better OOS and worse IS — that is instability, not a refinement to keep.\n")
        spec.append("The discretionary book has **no discernable profitable setup**. Later lots (after 2026-07-07) lost $363. MU/NVDA-only was +$560 early and −$266 later. The only filter that stayed green on both sides is **no overnight**, and that is flattening, not an entry pattern (n=12 in the later window). Public calendars explain the overnight damage (MU 6/24 earnings through SOXS, brand-new CBRX/SKUU, Aug HBM tightness vs a MUZ short) but they do not create an entry recipe.\n")

    spec.append("## Explicit non-rules\n")
    spec.append("- Do not treat a high in-sample win rate as an edge.")
    spec.append("- Do not promote RSI thresholds to rules from a single bucket with tiny n.")
    spec.append("- Do not trade SOXL/SOXS overnight because 'it will come back'. Direxion products reset daily; reverse splits are decay made visible.")
    spec.append("- Do not fade a memory blowout with overnight inverse 3x *through the print*. After the 8-K, a 21-day fade ban is not supported on this book.")
    spec.append("- Do not overnight a 2x ETF that listed this month.")
    spec.append("- Do not treat KeyBanc / CoreWeave / SMCI headlines as a reason to *short* MU overnight.")
    spec.append("- Do not increase size after a loss.")
    spec.append("- Do not go live on this specification.\n")
    spec.append("## Review trigger\n")
    spec.append("Re-run `python run.py` after ≥30 additional closed lots. If OOS still fails usefulness, retire the bot research line and keep only the risk constraints for discretionary trading.\n")

    spec_path = REPORT_DIR / "TRADING_SYSTEM_SPEC.md"
    spec_path.write_text("\n".join(spec))

    # compact json for canvas
    canvas = {
        "n_closed": int(len(closed)),
        "total_pnl": float(closed["pnl"].sum()) if len(closed) else 0.0,
        "win_rate": float((closed["pnl"] > 0).mean()) if len(closed) else None,
        "expectancy_pct": float(closed["return_pct"].mean()) if len(closed) else None,
        "oos_ok": oos_ok,
        "oos_note": oos_note,
        "ml_skill": ml_skill,
        "open_lots": opens.to_dict(orient="records") if not opens.empty else [],
        "keep_filters": keep_filters,
    }
    (PROC_DIR / "canvas_payload.json").write_text(json.dumps(canvas, indent=2, default=str))
    return {"research": research_path, "spec": spec_path, "canvas": canvas}
