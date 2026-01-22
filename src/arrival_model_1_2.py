"""
src/arrival_model_1_2.py

Module 1.2 — Case Arrival Modeling.

This module learns the arrival rate of new cases from historical data and
provides a runtime sampler for the simulation engine.

Methodology:
    Non-homogeneous Poisson Process (NHPP) with Piecewise Constant Intensity.
    
    1.  **Granularity**: We calculate the mean arrival rate (lambda) for every 
        specific hour of the week (Month + Day-of-Week + Hour).
    2.  **Fallback**: If data is sparse for a specific month, we fall back to 
        a general Day-of-Week + Hour model.
    3.  **Sampling**: At runtime, we sample inter-arrival times exponentially 
        based on the current hour's intensity.

Usage:
    ap = ArrivalProcess(csv_path="data/bpi2017.csv", seed=42)
    next_time = ap.next_arrival_time(current_time)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


# ----------------------------
# Training Artifact (Cached)
# ----------------------------
@dataclass(frozen=True)
class ArrivalModel:
    """
    Stores the learned arrival intensities.
    
    Attributes:
        mu_mdh: Mean arrivals/hour for specific (Month, Day-of-Week, Hour).
        mu_dh:  Mean arrivals/hour for (Day-of-Week, Hour) - used as fallback.
        mu_global: Overall mean arrivals/hour - used as final fallback.
    """
    mu_mdh: pd.Series
    mu_dh: pd.Series
    mu_global: float

    def expected_mu_per_hour(self, ts: pd.Timestamp) -> float:
        """
        Retrieves the expected arrival rate (lambda) for a given timestamp.
        Uses a hierarchical lookup strategy: specific -> general -> global.
        """
        m, d, h = int(ts.month), int(ts.dayofweek), int(ts.hour)

        # 1. Try specific Month + Day + Hour
        key = (m, d, h)
        if key in self.mu_mdh.index:
            return float(self.mu_mdh.loc[key])

        # 2. Try general Day + Hour
        key2 = (d, h)
        if key2 in self.mu_dh.index:
            return float(self.mu_dh.loc[key2])

        # 3. Global average
        return float(self.mu_global)


def fit_arrival_model_from_csv(
    csv_path: str,
    case_col: str = "case:concept:name",
    ts_col: str = "time:timestamp",
) -> ArrivalModel:
    """
    Parses the event log and calculates arrival statistics.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[case_col, ts_col])

    # Determine case start time (earliest event per case)
    created_times = df.groupby(case_col)[ts_col].min().sort_values()

    # Aggregate arrivals into hourly buckets
    hourly_counts = (
        created_times.to_frame("created_time")
        .set_index("created_time")
        .resample("h")
        .size()
        .asfreq("h", fill_value=0)
    )

    hourly_df = hourly_counts.to_frame("y")
    idx = hourly_df.index
    hourly_df["month"] = idx.month
    hourly_df["dow"] = idx.dayofweek
    hourly_df["hour"] = idx.hour

    # Calculate average rates
    mu_mdh = hourly_df.groupby(["month", "dow", "hour"])["y"].mean()
    mu_dh = hourly_df.groupby(["dow", "hour"])["y"].mean()
    mu_global = float(hourly_df["y"].mean())

    return ArrivalModel(mu_mdh=mu_mdh, mu_dh=mu_dh, mu_global=mu_global)


# ----------------------------
# Runtime Sampler
# ----------------------------
class ArrivalProcess:
    """
    Runtime interface for generating arrival events.
    Handles loading the cached model and performing the NHPP sampling.
    """

    def __init__(
        self,
        csv_path: str,
        seed: int = 42,
        cache_path: Optional[str] = None,
    ):
        self.csv_path = str(csv_path)
        self.rng = np.random.default_rng(seed)

        # Default Cache Path: models/arrival_model_1_2.pkl
        if cache_path is None:
            project_root = Path(__file__).resolve().parents[1]  # project/
            models_dir = project_root / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            cache_path = str(models_dir / "arrival_model_1_2.pkl")

        self.cache_path = cache_path
        self.model = self._load_or_train()

    def _load_or_train(self) -> ArrivalModel:
        """Loads cached model if available, otherwise triggers training."""
        p = Path(self.cache_path)

        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                pass

        model = fit_arrival_model_from_csv(self.csv_path)

        try:
            joblib.dump(model, p)
        except Exception:
            pass

        return model

    def _lambda_per_second(self, ts: pd.Timestamp) -> float:
        """Converts hourly rate to per-second intensity."""
        mu = max(self.model.expected_mu_per_hour(ts), 0.0)
        return mu / 3600.0

    def next_arrival_time(self, t_now) -> pd.Timestamp:
        """
        Samples the next arrival time using the thinning algorithm (or stepwise constant approximation).
        
        Logic:
        1. Calculate lambda for the current hour.
        2. Sample a wait time 'w' from Exp(lambda).
        3. If 'w' fits in the current hour, return (t_now + w).
        4. If 'w' overshoots the hour, advance to the next hour and retry.
        """
        t = pd.Timestamp(t_now)

        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")

        while True:
            hour_end = t.floor("h") + pd.Timedelta(hours=1)
            remaining = (hour_end - t).total_seconds()

            lam = self._lambda_per_second(t)
            
            # If rate is 0 (e.g., night), skip to next hour
            if not np.isfinite(lam) or lam <= 0.0:
                t = hour_end
                continue

            w = float(self.rng.exponential(scale=1.0 / lam))

            if w < remaining:
                return t + pd.to_timedelta(w, unit="s")

            # Overshot the hour boundary; advance time and resample with new rate
            t = hour_end