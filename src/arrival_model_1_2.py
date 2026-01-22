"""
arrival_model_1_2.py

Module 1.2 — Case arrivals (BPIC2017).

Train once from the real CSV, cache the model, then 1.1 just queries it.

Advanced idea:
  Arrival rate depends on (month, day-of-week, hour).
  Inside each hour the rate is constant (piecewise constant intensity).

API used by 1.1:
    ap = ArrivalProcess(csv_path="bpi2017.csv", seed=42)
    t_next = ap.next_arrival_time(t_now)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


# ----------------------------
# Training artifact (cached)
# ----------------------------
@dataclass(frozen=True)
class ArrivalModel:
    # mean arrivals/hour for (month, dow, hour)
    mu_mdh: pd.Series
    # fallback mean arrivals/hour for (dow, hour)
    mu_dh: pd.Series
    # final fallback
    mu_global: float

    def expected_mu_per_hour(self, ts: pd.Timestamp) -> float:
        m, d, h = int(ts.month), int(ts.dayofweek), int(ts.hour)

        key = (m, d, h)
        if key in self.mu_mdh.index:
            return float(self.mu_mdh.loc[key])

        key2 = (d, h)
        if key2 in self.mu_dh.index:
            return float(self.mu_dh.loc[key2])

        return float(self.mu_global)


def fit_arrival_model_from_csv(
    csv_path: str,
    case_col: str = "case:concept:name",
    ts_col: str = "time:timestamp",
) -> ArrivalModel:
    df = pd.read_csv(csv_path, low_memory=False)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[case_col, ts_col])

    # case start time = earliest timestamp per case
    created_times = df.groupby(case_col)[ts_col].min().sort_values()

    # cases started per hour
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

    mu_mdh = hourly_df.groupby(["month", "dow", "hour"])["y"].mean()
    mu_dh = hourly_df.groupby(["dow", "hour"])["y"].mean()
    mu_global = float(hourly_df["y"].mean())

    return ArrivalModel(mu_mdh=mu_mdh, mu_dh=mu_dh, mu_global=mu_global)


# ----------------------------
# Runtime sampler for 1.1
# ----------------------------
class ArrivalProcess:
    def __init__(
        self,
        csv_path: str,
        seed: int = 42,
        cache_path: Optional[str] = None,
    ):
        self.csv_path = str(csv_path)
        self.rng = np.random.default_rng(seed)

        # ✅ Default cache location: models/arrival_model_1_2.pkl
        if cache_path is None:
            project_root = Path(__file__).resolve().parents[1]  # project/
            models_dir = project_root / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            cache_path = str(models_dir / "arrival_model_1_2.pkl")

        self.cache_path = cache_path
        self.model = self._load_or_train()

    def _load_or_train(self) -> ArrivalModel:
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
        mu = max(self.model.expected_mu_per_hour(ts), 0.0)
        return mu / 3600.0

    def next_arrival_time(self, t_now) -> pd.Timestamp:
        t = pd.Timestamp(t_now)

        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")

        # sample waiting time inside current hour; if it doesn't fit, move to next hour
        while True:
            hour_end = t.floor("h") + pd.Timedelta(hours=1)
            remaining = (hour_end - t).total_seconds()

            lam = self._lambda_per_second(t)
            if not np.isfinite(lam) or lam <= 0.0:
                t = hour_end
                continue

            w = float(self.rng.exponential(scale=1.0 / lam))

            if w < remaining:
                return t + pd.to_timedelta(w, unit="s")

            t = hour_end
