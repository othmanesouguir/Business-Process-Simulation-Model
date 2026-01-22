"""
TRAINING SCRIPT for Task 1.3 (Processing Time) - OPTIMIZED

Key ideas:
1) Train once from the historical log
2) Save a single .pkl artifact (basic distributions + advanced ML models)
3) The simulation (1.1) only loads that artifact and queries durations

This version supports BOTH:
- CSV input (recommended, fast)
- XES input (kept as fallback)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

# Keep pm4py for XES fallback (optional)
try:
    import pm4py
except ImportError:
    pm4py = None

# LightGBM is only needed for advanced ML training
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ============================================================
# PATHS (project-safe)
# ============================================================
# This script lives in: project/src/
# CSV lives in:        project/data/bpi2017.csv
# Artifacts go to:     project/models/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = str(DATA_DIR / "bpi2017.csv")
MODEL_PATH = str(MODELS_DIR / "processing_model_advanced.pkl")


# ============================================================
# 1) Helper: lifecycle stats
# ============================================================
def analyze_lifecycle_transitions(df: pd.DataFrame):
    """Basic sanity prints to understand what transitions exist."""
    if "lifecycle:transition" not in df.columns:
        print("⚠️ No lifecycle:transition column found")
        return

    vc = df["lifecycle:transition"].astype(str).value_counts().head(20)
    print("\nLifecycle transition counts (top 20):")
    print(vc)


# ============================================================
# 2) Duration extraction (start/resume/suspend/complete)
# ============================================================
def calculate_net_durations_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes NET duration for each (case, activity, resource) segment.

    Works with transitions:
      start/resume -> work begins
      suspend      -> work pauses
      complete     -> work ends

    If something ends without perfect matching transitions,
    the code tries to handle it safely.
    """
    print("\nCalculating NET duration...")

    required_cols = ["case:concept:name", "concept:name", "time:timestamp"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "lifecycle:transition" not in df.columns:
        raise ValueError("Missing lifecycle:transition column")

    # Sort for correct sequencing
    df = df.sort_values(["case:concept:name", "time:timestamp"]).copy()

    WORK_TRANSITIONS = {"start", "resume", "suspend", "complete"}

    # Filter to only relevant transitions
    dfx = df[df["lifecycle:transition"].astype(str).isin(WORK_TRANSITIONS)].copy()
    if len(dfx) == 0:
        print("⚠️ No work transitions found (start/resume/suspend/complete).")
        return pd.DataFrame(
            columns=["case:concept:name", "concept:name", "org:resource", "net_duration_sec"]
        )

    # Make sure resource exists
    if "org:resource" not in dfx.columns:
        dfx["org:resource"] = ""

    records = []

    for case_id, g in dfx.groupby("case:concept:name"):
        current_activity = None
        current_resource = None
        last_start_time = None
        net_work = 0.0
        working = False

        for _, row in g.iterrows():
            act = str(row["concept:name"])
            res = str(row["org:resource"]) if pd.notna(row["org:resource"]) else ""
            tr = str(row["lifecycle:transition"])
            ts = row["time:timestamp"]

            if tr in {"start", "resume"}:
                current_activity = act
                current_resource = res
                last_start_time = ts
                working = True

            elif tr == "suspend":
                if working and current_activity == act and last_start_time is not None:
                    net_work += max(0.0, (ts - last_start_time).total_seconds())
                    working = False
                    last_start_time = None

            elif tr == "complete":
                if working and current_activity == act and last_start_time is not None:
                    net_work += max(0.0, (ts - last_start_time).total_seconds())
                    working = False
                    last_start_time = None

                # Write record for this completed activity chunk
                if current_activity is not None:
                    records.append(
                        {
                            "case:concept:name": case_id,
                            "concept:name": current_activity,
                            "org:resource": current_resource,
                            "net_duration_sec": float(net_work),
                        }
                    )

                # Reset for next activity
                current_activity = None
                current_resource = None
                net_work = 0.0
                working = False

    out = pd.DataFrame(records)
    out = out[out["net_duration_sec"] >= 0.0]
    print(f"   Produced {len(out):,} duration samples")
    return out


# ============================================================
# 3) Basic distributions per activity
# ============================================================
def fit_basic_distributions(duration_df: pd.DataFrame, min_samples: int = 50) -> dict:
    """
    Fit a simple distribution per activity.
    Returns a dict: activity -> distribution info
    """
    print("\nFitting BASIC distributions per activity...")

    basic = {}

    if len(duration_df) == 0:
        return basic

    for act, g in duration_df.groupby("concept:name"):
        x = g["net_duration_sec"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        x = x[x > 0]

        if len(x) < min_samples:
            continue

        # Try a few reasonable distributions
        candidates = []

        # lognorm
        try:
            shape, loc, scale = stats.lognorm.fit(x, floc=0)
            ll = np.sum(stats.lognorm.logpdf(x, shape, loc=loc, scale=scale))
            candidates.append(("lognorm", (shape, loc, scale), ll))
        except Exception:
            pass

        # gamma
        try:
            a, loc, scale = stats.gamma.fit(x, floc=0)
            ll = np.sum(stats.gamma.logpdf(x, a, loc=loc, scale=scale))
            candidates.append(("gamma", (a, loc, scale), ll))
        except Exception:
            pass

        # expon
        try:
            loc, scale = stats.expon.fit(x, floc=0)
            ll = np.sum(stats.expon.logpdf(x, loc=loc, scale=scale))
            candidates.append(("expon", (loc, scale), ll))
        except Exception:
            pass

        # norm
        try:
            mu, sigma = stats.norm.fit(x)
            ll = np.sum(stats.norm.logpdf(x, mu, sigma))
            candidates.append(("norm", (mu, sigma), ll))
        except Exception:
            pass

        if not candidates:
            continue

        dist_name, params, _ = max(candidates, key=lambda z: z[2])

        basic[act] = {
            "dist": dist_name,
            "params": params,
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "n": int(len(x)),
        }

    print(f"   Learned {len(basic)} basic distributions")
    return basic


# ============================================================
# 4) Feature engineering for ML (advanced)
# ============================================================
def merge_case_attributes(df_events: pd.DataFrame, duration_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge case-level attributes (case:*) into the duration samples.
    Fix: avoid duplicating 'case:concept:name' when doing reset_index().
    """
    case_cols = [c for c in df_events.columns if str(c).startswith("case:")]
    case_cols = [c for c in case_cols if c != "case:concept:name"]

    if len(case_cols) == 0:
        return duration_df

    case_attrs = df_events.groupby("case:concept:name", as_index=False)[case_cols].first()
    merged = duration_df.merge(case_attrs, on="case:concept:name", how="left")
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw columns to numeric feature matrix.
    Keep only a few meaningful ones.
    """
    out = df.copy()

    if "case:RequestedAmount" in out.columns:
        out["case:RequestedAmount"] = pd.to_numeric(out["case:RequestedAmount"], errors="coerce")

    if "CreditScore" in out.columns:
        out["CreditScore"] = pd.to_numeric(out["CreditScore"], errors="coerce")

    if "NumberOfTerms" in out.columns:
        out["NumberOfTerms"] = pd.to_numeric(out["NumberOfTerms"], errors="coerce")

    for col in ["case:LoanGoal", "case:ApplicationType"]:
        if col in out.columns:
            out[col] = out[col].astype("category")

    return out


def prepare_training_data(merged_df: pd.DataFrame, target_col: str = "net_duration_sec"):
    """
    Build X and y for LightGBM.
    """
    df = merged_df.copy()
    df = engineer_features(df)

    y = df[target_col].astype(float).to_numpy()
    X = df.drop(columns=[target_col], errors="ignore")

    drop_cols = ["case:concept:name", "concept:name", "org:resource"]
    for c in drop_cols:
        if c in X.columns:
            X = X.drop(columns=[c])

    return X, y


import re
import numpy as np
import lightgbm as lgb

def _sanitize_feature_names(cols):
    """
    LightGBM does not allow JSON special characters in feature names.
    We replace anything not [a-zA-Z0-9_] with underscore.
    Also ensure uniqueness.
    """
    mapping = {}
    used = set()

    for c in cols:
        orig = str(c)
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", orig)
        safe = re.sub(r"_+", "_", safe).strip("_")

        if safe == "":
            safe = "feature"

        base = safe
        k = 1
        while safe in used:
            k += 1
            safe = f"{base}_{k}"

        used.add(safe)
        mapping[orig] = safe

    return mapping


def train_lgbm_quantile_models(X, y, quantiles=(0.1, 0.5, 0.9)):
    """
    Train LightGBM quantile models for given quantiles.
    ✅ Fix: sanitize feature names so LightGBM doesn't crash.
    """
    models = {}

    X_train = X.copy()

    # LightGBM can handle categories but keep it safe
    for col in X_train.columns:
        if str(X_train[col].dtype) == "category":
            X_train[col] = X_train[col].cat.codes.replace(-1, np.nan)

    # Fill missing numeric values
    median_fill = X_train.median(numeric_only=True).to_dict()
    X_train = X_train.fillna(median_fill)

    # ✅ sanitize feature names (IMPORTANT FIX)
    feature_name_map = _sanitize_feature_names(list(X_train.columns))
    X_train = X_train.rename(columns=feature_name_map)

    for q in quantiles:
        params = {
            "objective": "quantile",
            "alpha": q,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
        }

        dtrain = lgb.Dataset(X_train, label=y, feature_name=list(X_train.columns))
        model = lgb.train(params, dtrain, num_boost_round=200)

        models[q] = model

    # median_fill must match renamed columns too
    median_fill_renamed = {feature_name_map.get(k, k): v for k, v in median_fill.items()}

    return models, list(X_train.columns), median_fill_renamed, feature_name_map

# ============================================================
# 5) Main training pipeline (CSV or XES)
# ============================================================
def train_processing_model(
    log_path: str = CSV_PATH,
    model_path: str = MODEL_PATH,
):
    print("=" * 80)
    print("TRAINING PROCESSING TIME MODEL (Task 1.3)")
    print("=" * 80)
    print(f"Source: {log_path}")
    print(f"Target: {model_path}")

    print("\nLoading event log...")

    # Fast path: CSV
    if str(log_path).lower().endswith(".csv"):
        df = pd.read_csv(log_path, low_memory=False)
        print("   Loaded CSV directly")
    else:
        if pm4py is None:
            raise ImportError("pm4py not installed. Install pm4py or use CSV input.")
        log = pm4py.read_xes(log_path)
        df = pm4py.convert_to_dataframe(log)
        print("   Loaded XES via pm4py")

    print(f"   Loaded {len(df):,} events from {df['case:concept:name'].nunique():,} cases")

    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"], utc=True, format="mixed", errors="coerce"
    )
    df = df.dropna(subset=["time:timestamp"])

    analyze_lifecycle_transitions(df)

    duration_df = calculate_net_durations_optimized(df)

    basic_distributions = fit_basic_distributions(duration_df, min_samples=50)

    ADV_ACTIVITIES = [
        "W_Validate application",
        "W_Call after offers",
        "W_Assess potential fraud",
        "W_Assess potential fraud 2",
    ]

    advanced_models = {}

    if len(duration_df) > 0:
        merged = merge_case_attributes(df, duration_df)

        for act in ADV_ACTIVITIES:
            sub = merged[merged["concept:name"].astype(str) == act].copy()
            sub = sub[sub["net_duration_sec"] > 0]

            if len(sub) < 200:
                continue

            X, y = prepare_training_data(sub, target_col="net_duration_sec")
            models, feature_cols, median_fill, feature_name_map = train_lgbm_quantile_models(
                X, y, quantiles=(0.1, 0.5, 0.9)
            )

            advanced_models[act] = {
                "models": models,
                "feature_cols": feature_cols,
                "median_fill": median_fill,
                "feature_name_map": feature_name_map,  # ✅ store mapping
            }

    means = [v["mean"] for v in basic_distributions.values() if "mean" in v]
    fallback_sec = float(np.median(means)) if means else 600.0

    artifact = {
        "basic_distributions": basic_distributions,
        "advanced_models": advanced_models,
        "fallback_sec": fallback_sec,
        "trained_at_utc": datetime.utcnow().isoformat(),
        "source_log": str(log_path),
    }

    # ensure models dir exists
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    print("\nSaved model artifact:", model_path)
    print("Basic dists:", len(basic_distributions))
    print("Advanced ML models:", len(advanced_models))
    print("Fallback sec:", fallback_sec)


if __name__ == "__main__":
    train_processing_model(
        log_path=CSV_PATH,
        model_path=MODEL_PATH,
    )
