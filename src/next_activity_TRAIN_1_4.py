"""
next_activity_TRAIN_1_4.py

Task 1.4 — Next activity predictor (train once).
Learns:
  P(next | prev2, prev1)

Backoff:
  if (prev2, prev1) unseen -> P(next | prev1)
  if prev1 unseen -> global P(next)

Control-flow learning:
  keep only end transitions: complete / withdraw / ate_abort
"""

from __future__ import annotations

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any

try:
    import pm4py
except ImportError:
    pm4py = None


# -------------------------
# PATHS (project-safe)
# -------------------------
# This file lives in: project/src/
# We want:
#   CSV in: project/data/
#   models in: project/models/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# CONFIG
# -------------------------
CSV_PATH = str(DATA_DIR / "bpi2017.csv")  # ✅ correct
XES_PATH = None                           # optional fallback if you want XES

OUT_PKL = str(MODELS_DIR / "next_activity_bigram_model.pkl")          # ✅ models/
OUT_CSV = str(MODELS_DIR / "next_activity_bigram_summary.csv")        # ✅ models/

END_TRANSITIONS = {"complete", "withdraw", "ate_abort"}
ONLY_WORKFLOW_ACTIVITIES = False
MIN_CONTEXT_COUNT = 5

START_TOKEN = "__START__"


# -------------------------
# LOAD
# -------------------------
def load_log(xes_path: Optional[str], csv_path: Optional[str]) -> pd.DataFrame:
    use_csv = csv_path is not None and os.path.exists(csv_path)
    use_xes = xes_path is not None and os.path.exists(xes_path)

    if use_csv:
        df = pd.read_csv(csv_path, low_memory=False)
        source = f"CSV: {csv_path}"
    elif use_xes:
        if pm4py is None:
            raise ImportError("pm4py not installed. Install pm4py or provide CSV_PATH.")
        log = pm4py.read_xes(xes_path)
        df = pm4py.convert_to_dataframe(log)
        source = f"XES: {xes_path}"
    else:
        raise FileNotFoundError(
            f"Neither CSV_PATH nor XES_PATH exists. Check paths.\n"
            f"CSV_PATH={csv_path}\nXES_PATH={xes_path}"
        )

    needed = {"case:concept:name", "concept:name", "time:timestamp"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["_row_id"] = np.arange(len(df), dtype=np.int64)

    # important: mixed timestamps in BPIC exports
    df["time:timestamp"] = pd.to_datetime(
        df["time:timestamp"], errors="coerce", utc=True, format="mixed"
    )
    df = df.dropna(subset=["time:timestamp", "case:concept:name", "concept:name"]).copy()

    if "lifecycle:transition" not in df.columns:
        df["lifecycle:transition"] = ""
    df["lifecycle:transition"] = (
        df["lifecycle:transition"].astype(str).str.lower().str.strip()
    )

    df = df.sort_values(["case:concept:name", "time:timestamp", "_row_id"])
    print(
        f"Loaded | events={len(df)} | cases={df['case:concept:name'].nunique()} | source={source}"
    )
    return df


# -------------------------
# Build sequences for control-flow
# -------------------------
def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["case:concept:name", "concept:name", "time:timestamp", "lifecycle:transition", "_row_id"]
    w = df[cols].copy()

    w["concept:name"] = w["concept:name"].astype(str)
    w = w[w["lifecycle:transition"].isin(END_TRANSITIONS)].copy()

    if ONLY_WORKFLOW_ACTIVITIES:
        w = w[w["concept:name"].str.startswith("W_")].copy()

    w = w.sort_values(["case:concept:name", "time:timestamp", "_row_id"])
    return w


# -------------------------
# TRAIN
# -------------------------
def train_next_activity_model(df_events: pd.DataFrame):
    sequences = build_sequences(df_events)

    rows: List[Tuple[str, str, str]] = []
    for _, g in sequences.groupby("case:concept:name", sort=False):
        acts = g["concept:name"].tolist()
        if len(acts) < 1:
            continue

        # learn early transitions too
        acts = [START_TOKEN, START_TOKEN] + acts

        for i in range(2, len(acts)):
            prev2, prev1, nxt = acts[i - 2], acts[i - 1], acts[i]
            rows.append((prev2, prev1, nxt))

    triples = pd.DataFrame(rows, columns=["prev2", "prev1", "next"])
    if triples.empty:
        raise RuntimeError("No triples could be built (check filters).")

    # bigram counts
    c_bigram = triples.groupby(["prev2", "prev1", "next"]).size().rename("count").reset_index()
    c_context = triples.groupby(["prev2", "prev1"]).size().rename("context_count").reset_index()
    c_bigram = c_bigram.merge(c_context, on=["prev2", "prev1"], how="left")

    # filter rare contexts
    c_bigram = c_bigram[c_bigram["context_count"] >= MIN_CONTEXT_COUNT].copy()
    c_bigram["prob"] = c_bigram["count"] / c_bigram["context_count"]

    # unigram backoff
    c_uni = triples.groupby(["prev1", "next"]).size().rename("count").reset_index()
    c_uni_ctx = triples.groupby(["prev1"]).size().rename("context_count").reset_index()
    c_uni = c_uni.merge(c_uni_ctx, on=["prev1"], how="left")
    c_uni["prob"] = c_uni["count"] / c_uni["context_count"]

    # global backoff
    c_glob = triples["next"].value_counts().rename_axis("next").reset_index(name="count")
    c_glob["prob"] = c_glob["count"] / c_glob["count"].sum()

    # dict format for fast simulation
    bigram_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (p2, p1), gg in c_bigram.groupby(["prev2", "prev1"]):
        bigram_dict[(p2, p1)] = {
            "next": gg["next"].tolist(),
            "prob": gg["prob"].to_numpy(dtype=float),
            "n": int(gg["context_count"].iloc[0]),
        }

    unigram_dict: Dict[str, Dict[str, Any]] = {}
    for p1, gg in c_uni.groupby(["prev1"]):
        unigram_dict[p1] = {
            "next": gg["next"].tolist(),
            "prob": gg["prob"].to_numpy(dtype=float),
            "n": int(gg["context_count"].iloc[0]),
        }

    global_dict = {
        "next": c_glob["next"].tolist(),
        "prob": c_glob["prob"].to_numpy(dtype=float),
        "n": int(c_glob["count"].sum()),
    }

    model = {
        "type": "next_activity_bigram",
        "start_token": START_TOKEN,
        "end_transitions": sorted(list(END_TRANSITIONS)),
        "only_workflow_activities": ONLY_WORKFLOW_ACTIVITIES,
        "min_context_count": MIN_CONTEXT_COUNT,
        "bigram": bigram_dict,
        "unigram": unigram_dict,
        "global": global_dict,
    }
    return model, c_bigram, c_uni, c_glob


def save_summary(c_bigram: pd.DataFrame, out_csv: str):
    ctx_sizes = (
        c_bigram.groupby(["prev2", "prev1"])["next"]
        .nunique()
        .rename("nb_possible_next")
        .reset_index()
    )
    ctx_counts = (
        c_bigram.groupby(["prev2", "prev1"])["count"]
        .sum()
        .rename("context_count")
        .reset_index()
    )

    summary = ctx_sizes.merge(ctx_counts, on=["prev2", "prev1"], how="left")
    summary = summary.sort_values(
        ["nb_possible_next", "context_count"], ascending=[False, False]
    )
    summary.to_csv(out_csv, index=False)


if __name__ == "__main__":
    df = load_log(XES_PATH, CSV_PATH)
    model, c_bigram, c_uni, c_glob = train_next_activity_model(df)

    joblib.dump(model, OUT_PKL)
    print(f"Saved model: {OUT_PKL}")

    save_summary(c_bigram, OUT_CSV)
    print(f"Saved summary: {OUT_CSV}")
