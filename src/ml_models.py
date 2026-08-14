"""Walk-forward classifiers on historical entries. No shuffled CV."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROC_DIR, RANDOM_SEED, TABLE_DIR, WALK_FORWARD_OOS_FRAC

warnings.filterwarnings("ignore")

FEATURE_CANDIDATES = [
    "hold_minutes",  # NOT used as a model feature — duration unknown at entry
    "und_rsi5",
    "und_rsi14",
    "und_dist_ema20",
    "und_dist_vwap",
    "und_dist_sess_high",
    "und_rvol",
    "und_vol_roll",
    "und_session_ret",
    "und_gap_pct",
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "ret_1h",
    "ret_1d",
    "spy_session_ret",
    "soxx_session_ret",
    "sector_confirm",
    "extended_2atr",
    "trend_vs_chop",
    "und_ema_bull",
    "und_ema_bear",
    "und_above_vwap",
    "und_vwap_reclaim",
    "und_vwap_reject",
    "und_ema20_reclaim",
    "und_ema20_reject",
    "und_hh",
    "und_hl",
    "und_lh",
    "und_ll",
    "und_breakout",
    "und_breakdown",
    "und_breakout_fail",
    "und_orb",
    "und_large_candle",
    "und_vol_expand",
    "und_vol_contract",
    "economic_side",
    "overnight",  # known only if planned; exclude from live features
    "leverage",
]


ENTRY_FEATURES = [
    c
    for c in FEATURE_CANDIDATES
    if c not in {"hold_minutes", "overnight"}
]


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    cols = [c for c in ENTRY_FEATURES if c in df.columns]
    x = df[cols].copy()
    for c in x.columns:
        if x[c].dtype == bool:
            x[c] = x[c].astype(float)
        else:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    y = (df["pnl"] > 0).astype(int).to_numpy()
    return x, y, cols


def _metrics(y_true, proba) -> dict:
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true)
    out = {
        "n": int(len(y_true)),
        "base_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "auc": np.nan,
        "ap": np.nan,
        "brier": np.nan,
        "logloss": np.nan,
    }
    if len(y_true) < 8 or len(np.unique(y_true)) < 2:
        return out
    try:
        out["auc"] = float(roc_auc_score(y_true, proba))
        out["ap"] = float(average_precision_score(y_true, proba))
        out["brier"] = float(brier_score_loss(y_true, proba))
        out["logloss"] = float(log_loss(y_true, proba))
    except Exception:
        pass
    return out


def _models():
    models = {
        "logistic": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=0.5,
                        max_iter=400,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=4,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.05,
                        min_child_samples=12,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        random_state=RANDOM_SEED,
                        verbose=-1,
                    ),
                ),
            ]
        )
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.05,
                        min_child_weight=2,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except Exception:
        pass
    return models


def _importance(name: str, model, cols: list[str]) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        val = np.abs(clf.coef_[0])
        kind = "abs_coef"
    elif hasattr(clf, "feature_importances_"):
        val = clf.feature_importances_
        kind = "impurity"
    else:
        return pd.DataFrame()
    return pd.DataFrame({"model": name, "feature": cols, "importance": val, "kind": kind}).sort_values(
        "importance", ascending=False
    )


def walk_forward(trades: pd.DataFrame) -> dict:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = trades[trades["pnl"].notna()].sort_values("entry_time").reset_index(drop=True)
    x, y, cols = _xy(df)
    n = len(df)
    results = []
    importances = []
    oos_rows = []
    folds = []
    # expanding walk-forward: 3 folds
    cuts = [0.5, 0.65, 1.0 - WALK_FORWARD_OOS_FRAC]
    for i, train_frac in enumerate(cuts):
        split = max(20, int(n * train_frac))
        if n - split < 8:
            continue
        xtr, ytr = x.iloc[:split], y[:split]
        xte, yte = x.iloc[split:], y[split:]
        folds.append({"fold": i, "train_n": int(len(xtr)), "test_n": int(len(xte)), "train_frac": train_frac})
        for name, model in _models().items():
            model.fit(xtr, ytr)
            proba = model.predict_proba(xte)[:, 1]
            m = _metrics(yte, proba)
            m.update({"fold": i, "model": name, "split": "expanding_oos"})
            results.append(m)
            imp = _importance(name, model, cols)
            if not imp.empty:
                imp["fold"] = i
                importances.append(imp)
            if i == len(cuts) - 1:
                tmp = df.iloc[split:].copy()
                tmp["p_win"] = proba
                tmp["model"] = name
                oos_rows.append(tmp[["entry_time", "symbol", "pnl", "return_pct", "p_win", "model"]])

    # single chronological 70/30
    split = max(20, int(n * (1 - WALK_FORWARD_OOS_FRAC)))
    xtr, ytr = x.iloc[:split], y[:split]
    xte, yte = x.iloc[split:], y[split:]
    last_models = {}
    for name, model in _models().items():
        model.fit(xtr, ytr)
        last_models[name] = model
        proba = model.predict_proba(xte)[:, 1]
        m = _metrics(yte, proba)
        m.update({"fold": "chrono_70_30", "model": name, "split": "holdout_oos"})
        results.append(m)
        in_m = _metrics(ytr, model.predict_proba(xtr)[:, 1])
        in_m.update({"fold": "chrono_70_30", "model": name, "split": "train_is"})
        results.append(in_m)
        imp = _importance(name, model, cols)
        if not imp.empty:
            imp["fold"] = "chrono_70_30"
            importances.append(imp)

    res_df = pd.DataFrame(results)
    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    res_df.to_csv(TABLE_DIR / "ml_walkforward.csv", index=False)
    if not imp_df.empty:
        imp_df.to_csv(TABLE_DIR / "ml_feature_importance.csv", index=False)
    if oos_rows:
        pd.concat(oos_rows, ignore_index=True).to_csv(TABLE_DIR / "ml_oos_predictions.csv", index=False)
    summary = {
        "n_samples": n,
        "n_features": len(cols),
        "warning": "Sample size is small; treat AUC near 0.5 as no skill.",
        "holdout_oos": res_df[res_df["split"] == "holdout_oos"].to_dict(orient="records"),
    }
    (PROC_DIR / "ml_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"metrics": res_df, "importance": imp_df, "summary": summary}
