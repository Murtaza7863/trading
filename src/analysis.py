"""Statistical research on historical round-trips. Hypotheses are tested, not assumed."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .config import N_BOOTSTRAP, PROC_DIR, RANDOM_SEED, TABLE_DIR, TZ, WALK_FORWARD_OOS_FRAC
from .events import event_calendar, sources_table, tag_events

RNG = np.random.default_rng(RANDOM_SEED)


def _ci(values: np.ndarray, stat=np.mean) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    point = float(stat(values))
    if len(values) < 5:
        return point, np.nan, np.nan
    boots = []
    n = len(values)
    for _ in range(N_BOOTSTRAP):
        sample = values[RNG.integers(0, n, n)]
        boots.append(stat(sample))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _row(name: str, df: pd.DataFrame) -> dict:
    pnl = df["pnl"].to_numpy(dtype=float)
    ret = df["return_pct"].to_numpy(dtype=float)
    wins = pnl[np.isfinite(pnl) & (pnl > 0)]
    losses = pnl[np.isfinite(pnl) & (pnl <= 0)]
    exp_p, exp_lo, exp_hi = _ci(ret)
    pnl_p, pnl_lo, pnl_hi = _ci(pnl)
    n = int(np.isfinite(pnl).sum())
    nw = int(len(wins))
    nl = int(len(losses))
    win_rate = nw / n if n else np.nan
    avg_win = float(np.mean(wins)) if nw else np.nan
    avg_loss = float(np.mean(losses)) if nl else np.nan
    gross_win = float(np.sum(wins)) if nw else 0.0
    gross_loss = float(-np.sum(losses)) if nl else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else np.nan
    expectancy = float(np.nanmean(ret)) if n else np.nan
    return {
        "group": name,
        "n": n,
        "n_wins": nw,
        "n_losses": nl,
        "win_rate": win_rate,
        "total_pnl": float(np.nansum(pnl)),
        "expectancy_pct": expectancy,
        "expectancy_pct_ci_lo": exp_lo,
        "expectancy_pct_ci_hi": exp_hi,
        "avg_pnl": float(np.nanmean(pnl)) if n else np.nan,
        "avg_pnl_ci_lo": pnl_lo,
        "avg_pnl_ci_hi": pnl_hi,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_avg_loss": (avg_win / abs(avg_loss)) if avg_loss and np.isfinite(avg_loss) else np.nan,
        "profit_factor": pf,
        "median_return_pct": float(np.nanmedian(ret)) if n else np.nan,
        "median_hold_min": float(np.nanmedian(df["hold_minutes"])) if n else np.nan,
    }


def group_expectancy(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(col, dropna=False):
        rows.append(_row(str(key), g))
    out = pd.DataFrame(rows).sort_values("total_pnl", ascending=False)
    return out


def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(name: str, a: pd.Series, b: pd.Series, ha: str = "two-sided"):
        a = a.dropna()
        b = b.dropna()
        if len(a) < 5 or len(b) < 5:
            p = np.nan
            u = np.nan
        else:
            u, p = stats.mannwhitneyu(a, b, alternative=ha)
        rows.append(
            {
                "hypothesis": name,
                "n_a": int(len(a)),
                "n_b": int(len(b)),
                "mean_a": float(a.mean()) if len(a) else np.nan,
                "mean_b": float(b.mean()) if len(b) else np.nan,
                "median_a": float(a.median()) if len(a) else np.nan,
                "median_b": float(b.median()) if len(b) else np.nan,
                "mw_u": float(u) if p == p else np.nan,
                "p_value": float(p) if p == p else np.nan,
                "significant_0_05": bool(p < 0.05) if p == p else False,
            }
        )

    add(
        "short_hold(<60m) vs long_hold(>=1d) return",
        df.loc[df["hold_minutes"] < 60, "return_pct"],
        df.loc[df["hold_minutes"] >= 1440, "return_pct"],
        "greater",
    )
    add(
        "intraday vs overnight return",
        df.loc[~df["overnight"].astype(bool), "return_pct"],
        df.loc[df["overnight"].astype(bool), "return_pct"],
        "greater",
    )
    add(
        "MU/NVDA family vs SOXX family return",
        df.loc[df["family"].isin(["MU", "NVDA"]), "return_pct"],
        df.loc[df["family"] == "SOXX", "return_pct"],
        "greater",
    )
    add(
        "economic long vs short return",
        df.loc[df["economic_side"] == 1, "return_pct"],
        df.loc[df["economic_side"] == -1, "return_pct"],
    )
    add(
        "reentry_after_loss_30m vs not return",
        df.loc[df["reentry_after_loss_30m"].astype(bool), "return_pct"],
        df.loc[~df["reentry_after_loss_30m"].astype(bool), "return_pct"],
        "less",
    )
    add(
        "reentry_after_loss_same_family vs not return",
        df.loc[df["reentry_after_loss_same_family"].astype(bool), "return_pct"],
        df.loc[~df["reentry_after_loss_same_family"].astype(bool), "return_pct"],
        "less",
    )
    if "und_vol_expand" in df.columns:
        add(
            "vol expansion vs contraction return",
            df.loc[df["und_vol_expand"].astype(bool), "return_pct"],
            df.loc[df["und_vol_contract"].astype(bool), "return_pct"],
        )
        add(
            "trend (ema aligned) vs chop return",
            df.loc[df["trend_vs_chop"] == 1, "return_pct"],
            df.loc[df["trend_vs_chop"] == 0, "return_pct"],
        )
        add(
            "extended_2atr vs not return",
            df.loc[df["extended_2atr"] == 1, "return_pct"],
            df.loc[df["extended_2atr"] != 1, "return_pct"],
            "less",
        )
        add(
            "breakout vs not return",
            df.loc[df["und_breakout"].astype(bool), "return_pct"],
            df.loc[~df["und_breakout"].astype(bool), "return_pct"],
        )
        add(
            "vwap above vs below return (economic long only)",
            df.loc[(df["economic_side"] == 1) & (df["und_above_vwap"].astype(bool)), "return_pct"],
            df.loc[(df["economic_side"] == 1) & (~df["und_above_vwap"].astype(bool)), "return_pct"],
        )
    if "event_any" in df.columns:
        add(
            "calendar-event lots vs rest return",
            df.loc[df["event_any"].astype(bool), "return_pct"],
            df.loc[~df["event_any"].astype(bool), "return_pct"],
            "less",
        )
        add(
            "event+overnight vs rest return",
            df.loc[df["event_overnight"].astype(bool), "return_pct"],
            df.loc[~df["event_overnight"].astype(bool), "return_pct"],
            "less",
        )
    return pd.DataFrame(rows)


def winner_loser_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number, bool]).copy()
    skip = {
        "lot_id",
        "campaign_id",
        "qty",
        "entry_price",
        "exit_price",
        "entry_fill_id",
        "exit_fill_id",
        "pnl",
        "return_pct",
        "entry_notional",
        "exit_notional",
        "leverage",
        "economic_side",
        "winner",
        "loser",
        "mae_pct",
        "mfe_pct",
        "max_unrealized_pnl_pct",
        "max_unrealized_loss_pct",
        "minutes_to_mfe",
        "minutes_to_mae",
        "und_close",
        "und_vwap",
        "und_ema9",
        "und_ema20",
        "und_ema50",
        "und_ema200",
        "und_atr14",
        "und_pdh",
        "und_pdl",
        "und_or_high",
        "und_or_low",
        "path_bars",
        "path_price_scale",
        "hold_minutes",
        "overnight",
        "intraday",
        "reentry_after_loss_30m",
        "reentry_after_loss_same_family",
        "prev_was_loss",
        "minutes_since_prev_exit",
        "minutes_to_mfe",
        "minutes_to_mae",
        "event_earnings",
        "event_fomc",
        "event_new_product",
        "event_fade_memory_post_mu_earn",
        "event_hbm_squeeze",
        "event_any",
        "event_overnight",
        "product_age_days",
    }
    cols = [c for c in numeric.columns if c not in skip]
    rows = []
    w = df["winner"].astype(bool)
    for col in cols:
        a = pd.to_numeric(df.loc[w, col], errors="coerce")
        b = pd.to_numeric(df.loc[~w, col], errors="coerce")
        a = a.dropna()
        b = b.dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        try:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        rows.append(
            {
                "feature": col,
                "winner_mean": float(a.mean()),
                "loser_mean": float(b.mean()),
                "winner_median": float(a.median()),
                "loser_median": float(b.median()),
                "p_value": float(p),
                "significant_0_05": bool(p < 0.05),
                "direction": "higher_in_winners" if a.mean() > b.mean() else "higher_in_losers",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("p_value")
    return out


def rsi_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if "und_rsi14" not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["rsi14_bucket"] = pd.cut(
        tmp["und_rsi14"],
        bins=[0, 30, 40, 50, 60, 70, 100],
        labels=["0-30", "30-40", "40-50", "50-60", "60-70", "70-100"],
    )
    tmp["rsi5_bucket"] = pd.cut(
        tmp["und_rsi5"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
    )
    a = group_expectancy(tmp.dropna(subset=["rsi14_bucket"]), "rsi14_bucket")
    a["kind"] = "rsi14"
    b = group_expectancy(tmp.dropna(subset=["rsi5_bucket"]), "rsi5_bucket")
    b["kind"] = "rsi5"
    return pd.concat([a, b], ignore_index=True)


def ema_vwap_buckets(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["ema_cfg"] = np.select(
        [tmp.get("und_ema_bull", False), tmp.get("und_ema_bear", False)],
        ["bull_aligned", "bear_aligned"],
        default="mixed_chop",
    )
    tmp["vwap_rel"] = np.where(tmp.get("und_above_vwap", False), "above_vwap", "below_vwap")
    tmp["rvol_bucket"] = pd.cut(
        pd.to_numeric(tmp.get("und_rvol"), errors="coerce"),
        bins=[0, 0.8, 1.2, 2, 100],
        labels=["low<0.8", "avg_0.8-1.2", "high_1.2-2", "very_high>2"],
    )
    parts = [
        group_expectancy(tmp, "ema_cfg"),
        group_expectancy(tmp, "vwap_rel"),
        group_expectancy(tmp.dropna(subset=["rvol_bucket"]), "rvol_bucket"),
    ]
    return pd.concat(parts, ignore_index=True)


def stop_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Counterfactual hard stop from entry using MAE/MFE timestamps when available."""
    rows = []
    stops_pct = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    for s in stops_pct:
        realized = []
        stopped_winners = 0
        capped_losers = 0
        n_stop = 0
        for _, tr in df.iterrows():
            ret = tr["return_pct"]
            mae = tr.get("mae_pct", np.nan)
            mfe = tr.get("mfe_pct", np.nan)
            t_mae = tr.get("minutes_to_mae", np.nan)
            t_mfe = tr.get("minutes_to_mfe", np.nan)
            if not np.isfinite(ret):
                continue
            hit_stop = np.isfinite(mae) and mae <= -s
            if hit_stop:
                n_stop += 1
                # First-touch by time. Equal timestamps (same bar) are unordered.
                stop_first = True
                if np.isfinite(t_mae) and np.isfinite(t_mfe) and np.isfinite(mfe) and mfe > 0:
                    if t_mae < t_mfe:
                        stop_first = True
                    elif t_mfe < t_mae:
                        stop_first = False
                    else:
                        stop_first = False
                if stop_first:
                    realized.append(-s)
                    if ret > 0:
                        stopped_winners += 1
                    elif ret < -s:
                        capped_losers += 1
                else:
                    realized.append(ret)
            else:
                realized.append(ret)
        arr = np.array(realized, dtype=float)
        rows.append(
            {
                "stop_pct": s,
                "n": int(len(arr)),
                "expectancy_pct": float(arr.mean()) if len(arr) else np.nan,
                "total_pnl_proxy_pct": float(arr.sum()) if len(arr) else np.nan,
                "win_rate": float((arr > 0).mean()) if len(arr) else np.nan,
                "n_stopped": n_stop,
                "winners_stopped": stopped_winners,
                "losers_capped": capped_losers,
                "profit_factor": _pf(arr),
            }
        )
    return pd.DataFrame(rows)


def _pf(arr: np.ndarray) -> float:
    gw = arr[arr > 0].sum()
    gl = -arr[arr <= 0].sum()
    return float(gw / gl) if gl > 0 else np.nan


def takeprofit_grid(df: pd.DataFrame, stop_pct: float | None = None) -> pd.DataFrame:
    rows = []
    tps = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0]
    for t in tps:
        realized = []
        for _, tr in df.iterrows():
            ret = tr["return_pct"]
            mae = tr.get("mae_pct", np.nan)
            mfe = tr.get("mfe_pct", np.nan)
            t_mae = tr.get("minutes_to_mae", np.nan)
            t_mfe = tr.get("minutes_to_mfe", np.nan)
            if not np.isfinite(ret):
                continue
            hit_tp = np.isfinite(mfe) and mfe >= t
            hit_stop = stop_pct is not None and np.isfinite(mae) and mae <= -stop_pct
            if hit_tp and hit_stop and np.isfinite(t_mae) and np.isfinite(t_mfe):
                if t_mfe < t_mae:
                    realized.append(t)
                elif t_mae < t_mfe:
                    realized.append(-stop_pct)
                else:
                    realized.append(ret)
            elif hit_stop and (not hit_tp or (np.isfinite(t_mae) and np.isfinite(t_mfe) and t_mae < t_mfe)):
                realized.append(-stop_pct)
            elif hit_tp:
                realized.append(t)
            else:
                realized.append(ret)
        arr = np.array(realized, dtype=float)
        rows.append(
            {
                "takeprofit_pct": t,
                "stop_pct": stop_pct,
                "n": int(len(arr)),
                "expectancy_pct": float(arr.mean()) if len(arr) else np.nan,
                "win_rate": float((arr > 0).mean()) if len(arr) else np.nan,
                "profit_factor": _pf(arr),
            }
        )
    return pd.DataFrame(rows)


def hold_decay(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cap in [5, 10, 15, 30, 45, 60, 90, 120, 240, 480, 1440]:
        # If MFE/MAE path exists, we cannot know exact mark at cap without bars.
        # Use actual trades that exited by cap vs those that held longer.
        g = df[df["hold_minutes"] <= cap]
        h = df[df["hold_minutes"] > cap]
        rows.append(
            {
                "max_hold_min": cap,
                **{f"exited_by_cap_{k}": v for k, v in _row("x", g).items() if k != "group"},
                "n_held_longer": int(h["pnl"].notna().sum()),
                "held_longer_expectancy_pct": float(h["return_pct"].mean()) if len(h) else np.nan,
                "held_longer_total_pnl": float(h["pnl"].sum()) if len(h) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def event_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Dollar damage sitting on publicly dated events vs the rest of the book."""
    flags = [
        ("earnings_window", df["event_earnings"].astype(bool)),
        ("fomc_window", df["event_fomc"].astype(bool)),
        ("new_2x_product_<14d", df["event_new_product"].astype(bool)),
        ("fade_inverse_21d_after_MU_earn", df["event_fade_memory_post_mu_earn"].astype(bool)),
        ("hbm_squeeze_week_context", df["event_hbm_squeeze"].astype(bool)),
        ("any_calendar_event", df["event_any"].astype(bool)),
        ("calendar_event_and_overnight", df["event_overnight"].astype(bool)),
        (
            "overnight_without_calendar_event",
            df["overnight"].astype(bool) & ~df["event_any"].astype(bool),
        ),
        ("rest_no_calendar_event", ~df["event_any"].astype(bool)),
    ]
    book = float(df["pnl"].sum())
    rows = []
    for name, mask in flags:
        row = _row(name, df[mask])
        row["share_of_book_pnl"] = (row["total_pnl"] / book) if book else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def overlay_rules(df: pd.DataFrame, camps: pd.DataFrame) -> pd.DataFrame:
    """Expectancy if historical entries are filtered / modified."""
    rows = []
    rows.append(_row("baseline_all_lots", df))
    rows.append(_row("no_overnight", df[~df["overnight"].astype(bool)]))
    rows.append(_row("hold_lt_60m", df[df["hold_minutes"] < 60]))
    rows.append(_row("hold_lt_240m", df[df["hold_minutes"] < 240]))
    rows.append(_row("mu_nvda_only", df[df["family"].isin(["MU", "NVDA"])]))
    rows.append(_row("exclude_soxx", df[df["family"] != "SOXX"]))
    rows.append(_row("no_reentry_30m_after_loss", df[~df["reentry_after_loss_30m"].astype(bool)]))
    rows.append(
        _row(
            "no_reentry_30m_after_loss_same_family",
            df[~df["reentry_after_loss_same_family"].astype(bool)],
        )
    )
    if not camps.empty and "averaged_down" in camps.columns:
        bad_ids = set(camps.loc[camps["averaged_down"].astype(bool), "campaign_id"])
        rows.append(_row("no_averaging_down_campaigns", df[~df["campaign_id"].isin(bad_ids)]))
        rows.append(_row("averaging_down_campaigns_only", df[df["campaign_id"].isin(bad_ids)]))
    if "und_ema_bull" in df.columns:
        aligned = df[
            ((df["economic_side"] == 1) & df["und_ema_bull"].astype(bool))
            | ((df["economic_side"] == -1) & df["und_ema_bear"].astype(bool))
        ]
        rows.append(_row("ema_aligned_with_economic_side", aligned))
        vwap_ok = df[
            ((df["economic_side"] == 1) & df["und_above_vwap"].astype(bool))
            | ((df["economic_side"] == -1) & ~df["und_above_vwap"].astype(bool))
        ]
        rows.append(_row("vwap_with_economic_side", vwap_ok))
        rows.append(_row("not_extended_2atr", df[df["extended_2atr"] != 1]))
    if "event_any" in df.columns:
        rows.append(_row("no_earnings_window", df[~df["event_earnings"].astype(bool)]))
        rows.append(
            _row(
                "no_earnings_overnight",
                df[~(df["event_earnings"].astype(bool) & df["overnight"].astype(bool))],
            )
        )
        rows.append(_row("no_new_2x_product", df[~df["event_new_product"].astype(bool)]))
        rows.append(_row("no_fade_memory_post_mu_earn", df[~df["event_fade_memory_post_mu_earn"].astype(bool)]))
        rows.append(
            _row(
                "no_fomc_overnight",
                df[~(df["event_fomc"].astype(bool) & df["overnight"].astype(bool))],
            )
        )
        rows.append(_row("no_event_overnight", df[~df["event_overnight"].astype(bool)]))
    return pd.DataFrame(rows)


def _user_filter_masks(df: pd.DataFrame, camps: pd.DataFrame) -> dict[str, pd.Series]:
    idx = df.index
    masks = {
        "all_lots": pd.Series(True, index=idx),
        "no_overnight": ~df["overnight"].astype(bool),
        "mu_nvda_only": df["family"].isin(["MU", "NVDA"]),
        "exclude_soxx": df["family"] != "SOXX",
        "not_morning": df["entry_session"] != "morning",
        "no_reentry_family": ~df["reentry_after_loss_same_family"].astype(bool),
    }
    if not camps.empty and "averaged_down" in camps.columns:
        bad = set(camps.loc[camps["averaged_down"].astype(bool), "campaign_id"])
        masks["no_averaging_down"] = ~df["campaign_id"].isin(bad)
    if "extended_2atr" in df.columns:
        masks["not_extended_2atr"] = df["extended_2atr"] != 1
    if "trend_vs_chop" in df.columns:
        masks["chop_only"] = df["trend_vs_chop"] != 1
    if "und_ema_bear" in df.columns:
        masks["not_ema_bear"] = ~df["und_ema_bear"].astype(bool)
    masks["same_session_mu_nvda"] = masks["no_overnight"] & masks["mu_nvda_only"]
    masks["same_session_mu_nvda_not_morning"] = (
        masks["same_session_mu_nvda"] & masks["not_morning"]
    )
    if "event_any" in df.columns:
        masks["no_earnings_window"] = ~df["event_earnings"].astype(bool)
        masks["no_earnings_overnight"] = ~(
            df["event_earnings"].astype(bool) & df["overnight"].astype(bool)
        )
        masks["no_new_2x_product"] = ~df["event_new_product"].astype(bool)
        masks["no_fade_memory_post_mu_earn"] = ~df["event_fade_memory_post_mu_earn"].astype(
            bool
        )
        masks["no_event_overnight"] = ~df["event_overnight"].astype(bool)
    return masks


def chronological_user_filters(df: pd.DataFrame, camps: pd.DataFrame) -> pd.DataFrame:
    """Apply discretionary filters on early lots vs later lots. Same cut as the engine."""
    closed = df[df["pnl"].notna()].copy()
    if closed.empty or "entry_time" not in closed.columns:
        return pd.DataFrame()
    start = pd.Timestamp(closed["entry_time"].min())
    end = pd.Timestamp(closed["exit_time"].max())
    if start.tzinfo is None:
        start = start.tz_localize(TZ)
    if end.tzinfo is None:
        end = end.tz_localize(TZ)
    cut = start + (end - start) * (1 - WALK_FORWARD_OOS_FRAC)
    is_mask = closed["entry_time"] < cut
    rows = []
    for name, mask in _user_filter_masks(closed, camps).items():
        mask = mask.reindex(closed.index).fillna(False)
        for split, part in (("IS", closed[is_mask]), ("OOS", closed[~is_mask])):
            g = part[mask.loc[part.index]]
            row = _row(name, g)
            row["split"] = split
            row["cut"] = str(cut)
            rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(trades: pd.DataFrame, camps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    closed = trades[trades["pnl"].notna()].copy()
    closed = tag_events(closed)
    tables = {
        "overall": pd.DataFrame([_row("all_closed_lots", closed)]),
        "by_symbol": group_expectancy(closed, "symbol"),
        "by_family": group_expectancy(closed, "family"),
        "by_economic_side": group_expectancy(closed, "economic_side"),
        "by_overnight": group_expectancy(closed, "overnight"),
        "by_hold_bucket": group_expectancy(closed, "hold_bucket"),
        "by_entry_session": group_expectancy(closed, "entry_session"),
        "by_outcome": group_expectancy(closed, "outcome"),
        "by_reentry": group_expectancy(closed, "reentry_after_loss_30m"),
        "by_reentry_family": group_expectancy(closed, "reentry_after_loss_same_family"),
        "hypotheses": hypothesis_tests(closed),
        "winner_loser_features": winner_loser_features(closed),
        "rsi_buckets": rsi_buckets(closed),
        "ema_vwap_rvol": ema_vwap_buckets(closed),
        "stop_grid": stop_grid(closed),
        "takeprofit_no_stop": takeprofit_grid(closed, None),
        "hold_decay": hold_decay(closed),
        "rule_overlays": overlay_rules(closed, camps),
        "user_filters_chrono": chronological_user_filters(closed, camps),
        "event_attribution": event_attribution(closed),
        "event_calendar": event_calendar(),
        "event_sources": sources_table(),
        "by_event_earnings": group_expectancy(closed, "event_earnings"),
        "by_event_new_product": group_expectancy(closed, "event_new_product"),
        "by_event_fade_memory": group_expectancy(closed, "event_fade_memory_post_mu_earn"),
        "event_lots": closed.loc[
            closed["event_any"].astype(bool) | closed["event_hbm_squeeze"].astype(bool),
            [
                "symbol",
                "family",
                "entry_time",
                "exit_time",
                "overnight",
                "economic_side",
                "pnl",
                "return_pct",
                "event_earnings",
                "event_earnings_name",
                "event_fomc",
                "event_new_product",
                "product_age_days",
                "event_fade_memory_post_mu_earn",
                "event_hbm_squeeze",
                "event_any",
            ],
        ].sort_values("pnl"),
        "campaigns_by_avgdown": group_expectancy(
            camps[camps["pnl"].notna()], "averaged_down"
        )
        if not camps.empty
        else pd.DataFrame(),
    }
    # stop + TP joint using best stop from grid later in report
    if not tables["stop_grid"].empty:
        best_stop = float(tables["stop_grid"].sort_values("expectancy_pct", ascending=False).iloc[0]["stop_pct"])
        tables["takeprofit_with_best_is_stop"] = takeprofit_grid(closed, best_stop)
        tables["best_stop_note"] = pd.DataFrame(
            [{"best_stop_pct_by_in_sample_expectancy": best_stop, "warning": "selected on full sample; confirm OOS"}]
        )
    for name, tbl in tables.items():
        if isinstance(tbl, pd.DataFrame) and not tbl.empty:
            tbl.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    summary = {
        "n_closed_lots": int(len(closed)),
        "n_campaigns": int(camps["pnl"].notna().sum()) if not camps.empty else 0,
        "total_pnl": float(closed["pnl"].sum()),
        "win_rate": float((closed["pnl"] > 0).mean()) if len(closed) else None,
        "expectancy_pct": float(closed["return_pct"].mean()) if len(closed) else None,
        "catastrophic_pnl": float(
            closed.loc[closed["outcome"] == "catastrophic_loser", "pnl"].sum()
        ),
        "catastrophic_n": int((closed["outcome"] == "catastrophic_loser").sum()),
        "overnight_pnl": float(closed.loc[closed["overnight"].astype(bool), "pnl"].sum()),
        "intraday_pnl": float(closed.loc[~closed["overnight"].astype(bool), "pnl"].sum()),
        "event_overnight_pnl": float(closed.loc[closed["event_overnight"].astype(bool), "pnl"].sum()),
        "event_overnight_n": int(closed["event_overnight"].astype(bool).sum()),
        "hbm_squeeze_pnl": float(closed.loc[closed["event_hbm_squeeze"].astype(bool), "pnl"].sum()),
    }
    closed.to_csv(PROC_DIR / "trades_event_tagged.csv", index=False)
    (PROC_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    tables["summary_json"] = pd.DataFrame([summary])
    tables["summary_json"].to_csv(TABLE_DIR / "summary.json.csv", index=False)
    return tables
