"""
src/resource_availability_1_5.py

Module 1.5 — Resource Availability Modeling.

This module learns the working schedules of resources from historical event logs.
It answers the question: "Which resources are working at timestamp T?"

Modes:
- 'basic':  Determines availability based on a 2-week cycle (Even/Odd ISO weeks).
            Uses 1-hour time buckets.
- 'advanced': Determines availability based on Month + Weekday patterns.
            Uses 2-hour time buckets.
            Adds an 'Absence Filter': Resources must have at least one event on 
            a specific calendar day to be considered 'present'.

Usage:
    model = ResourceAvailabilityModel(csv_path="data/bpi2017.csv", mode="advanced")
    available_users = model.get_available_resources(current_simulation_time)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd


def _load_and_repair_minimal(csv_path: str) -> pd.DataFrame:
    """
    Loads only the necessary columns for availability modeling.
    Repairs broken rows where the resource ID is merged into the Action column.
    """
    df = pd.read_csv(
        csv_path,
        usecols=["Action", "org:resource", "time:timestamp"],
        low_memory=False,
    )

    # Detect and fix malformed rows (specific to BPI 2017 dataset quirks)
    broken = df["org:resource"].isna() & df["Action"].astype(str).str.contains(",", regex=False)
    if broken.any():
        # Split the comma-separated string to recover lost columns
        parts = df.loc[broken, "Action"].astype(str).str.split(",", n=18, expand=True)
        # Empirical mapping: index 1 -> org:resource, index 6 -> time:timestamp
        df.loc[broken, "org:resource"] = parts[1].values
        df.loc[broken, "time:timestamp"] = parts[6].values

    # Standardize timestamp and resource columns
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["time:timestamp", "org:resource"]).copy()
    df["org:resource"] = df["org:resource"].astype(str).str.strip()
    return df


@dataclass(frozen=True)
class AvailabilityArtifactBasic:
    """Storage container for the Basic Mode trained model."""
    tau: float
    # Key: (week_type [0=even, 1=odd], weekday [0-6], hour [0-23]) -> List[Resource IDs]
    avail_index: Dict[Tuple[int, int, int], List[str]] 


@dataclass(frozen=True)
class AvailabilityArtifactAdvanced:
    """Storage container for the Advanced Mode trained model."""
    tau_month: float
    # Key: Date Object -> Set of resources active that day
    present_by_date: Dict[object, Set[str]] 
    # Key: (month [1-12], weekday [0-6], bucket2h [0-11]) -> List[Resource IDs]
    monthly_index: Dict[Tuple[int, int, int], List[str]] 


def _train_basic(df_min: pd.DataFrame, tau: float) -> AvailabilityArtifactBasic:
    """
    Trains the Basic model (2-week cycle).
    A resource is 'available' in a slot if their presence probability > tau.
    """
    ts = df_min["time:timestamp"]
    iso = ts.dt.isocalendar()

    tmp = pd.DataFrame(
        {
            "org:resource": df_min["org:resource"].astype("string"),
            "iso_year": iso["year"].astype(np.int32),
            "iso_week": iso["week"].astype(np.int16),
            "weekday": ts.dt.weekday.astype(np.int8),
            "hour": ts.dt.hour.astype(np.int8),
        }
    )

    tmp["week_id"] = (tmp["iso_year"] * 100 + tmp["iso_week"]).astype(np.int32)
    tmp["week_type"] = (tmp["iso_week"] % 2).astype(np.int8)  # 0=even, 1=odd

    # Total unique weeks of each type in the dataset
    week_counts = (
        tmp[["week_id", "week_type"]]
        .drop_duplicates()
        .groupby("week_type", observed=True)["week_id"]
        .nunique()
        .sort_index()
    )

    # Count how many weeks the resource actually worked in each specific slot
    presence = tmp[["org:resource", "week_type", "week_id", "weekday", "hour"]].drop_duplicates()
    worked = (
        presence.groupby(["org:resource", "week_type", "weekday", "hour"], observed=True)["week_id"]
        .nunique()
        .reset_index(name="worked_weeks")
    )

    worked["total_weeks"] = worked["week_type"].map(week_counts).astype(np.int16)
    worked["p_worked"] = worked["worked_weeks"] / worked["total_weeks"]
    worked["available"] = worked["p_worked"] >= float(tau)

    # Build the lookup index
    avail_index: Dict[Tuple[int, int, int], List[str]] = {}
    for (wt, wd, hr), sub in worked[worked["available"]].groupby(
        ["week_type", "weekday", "hour"], observed=True
    ):
        avail_index[(int(wt), int(wd), int(hr))] = sub["org:resource"].astype(str).tolist()

    return AvailabilityArtifactBasic(tau=float(tau), avail_index=avail_index)


def _train_advanced(df_min: pd.DataFrame, tau_month: float) -> AvailabilityArtifactAdvanced:
    """
    Trains the Advanced model (Month + 2h Buckets + Daily Presence check).
    """
    ts = df_min["time:timestamp"]

    tmp = pd.DataFrame(
        {
            "org:resource": df_min["org:resource"].astype("string"),
            "date": ts.dt.date,
            "month": ts.dt.month.astype(np.int8),
            "weekday": ts.dt.weekday.astype(np.int8),
            "bucket2h": (ts.dt.hour // 2).astype(np.int8),
        }
    )

    # 1. Daily Presence Map (Who was physically present on Date X?)
    present_days = tmp.drop_duplicates(subset=["org:resource", "date", "month", "weekday"])
    present_by_date = (
        present_days.groupby("date", observed=True)["org:resource"]
        .apply(lambda s: set(s.astype(str).tolist()))
        .to_dict()
    )

    # 2. Denominator: Total days resource was present in that Month/Weekday
    denom_present = (
        present_days.groupby(["org:resource", "month", "weekday"], observed=True)["date"]
        .nunique()
        .rename("present_days")
        .reset_index()
    )

    # 3. Numerator: How many times did they appear in this specific 2h bucket?
    presence_day_bucket = tmp.drop_duplicates(
        subset=["org:resource", "date", "month", "weekday", "bucket2h"]
    )
    num_bucket = (
        presence_day_bucket.groupby(["org:resource", "month", "weekday", "bucket2h"], observed=True)["date"]
        .nunique()
        .rename("days_in_bucket")
        .reset_index()
    )

    # 4. Calculate Probability
    monthly_model = num_bucket.merge(
        denom_present,
        on=["org:resource", "month", "weekday"],
        how="left",
    )
    monthly_model["p_worked"] = monthly_model["days_in_bucket"] / monthly_model["present_days"]
    monthly_model["available"] = monthly_model["p_worked"] >= float(tau_month)

    # 5. Build Lookup Index
    monthly_index = (
        monthly_model[monthly_model["available"]]
        .groupby(["month", "weekday", "bucket2h"], observed=True)["org:resource"]
        .apply(lambda s: s.astype(str).tolist())
        .to_dict()
    )

    return AvailabilityArtifactAdvanced(
        tau_month=float(tau_month),
        present_by_date=present_by_date,
        monthly_index=monthly_index,
    )


class ResourceAvailabilityModel:
    """
    Public Interface for Resource Availability.
    Handles caching (loading/saving .pkl files) and runtime querying.
    """

    def __init__(
        self,
        csv_path: str,
        mode: str = "basic",
        *,
        tau: float = 0.50,
        tau_month: float = 0.50,
        cache_path: Optional[str] = None,
        year_filter: int = 2016,
    ):
        self.csv_path = str(csv_path)
        self.mode = mode.lower().strip()
        if self.mode not in {"basic", "advanced"}:
            raise ValueError("Mode must be 'basic' or 'advanced'")

        self.tau = float(tau)
        self.tau_month = float(tau_month)
        self.year_filter = int(year_filter)

        # Default cache path: models/resource_availability_model_1_5_{mode}.pkl
        if cache_path is None:
            project_root = Path(__file__).resolve().parents[1]
            models_dir = project_root / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            cache_path = str(models_dir / f"resource_availability_model_1_5_{self.mode}.pkl")

        self.cache_path = cache_path
        self.artifact = self._load_or_train()

    def _load_or_train(self):
        """Loads cached model if available, otherwise trains from scratch."""
        p = Path(self.cache_path)
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                pass  # Fallback to retraining if load fails

        print(f"[ResourceAvailability] Training new model (Mode: {self.mode})...")
        df = _load_and_repair_minimal(self.csv_path)

        # Train on 2016 data only (Stable period)
        start = pd.Timestamp(f"{self.year_filter}-01-01", tz="UTC")
        end = pd.Timestamp(f"{self.year_filter + 1}-01-01", tz="UTC")
        df = df[(df["time:timestamp"] >= start) & (df["time:timestamp"] < end)].copy()

        if self.mode == "basic":
            art = _train_basic(df, tau=self.tau)
        else:
            art = _train_advanced(df, tau_month=self.tau_month)

        # Save to cache
        try:
            joblib.dump(art, p)
            print(f"[ResourceAvailability] Model saved to {p}")
        except Exception:
            pass
        return art

    def get_available_resources(self, t) -> List[str]:
        """
        Returns a list of resources working at timestamp t.
        Used by the Simulation Engine to determine capacity.
        """
        ts = pd.Timestamp(t)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        if self.mode == "basic":
            week_type = int(ts.isocalendar().week) % 2
            weekday = int(ts.weekday())
            hour = int(ts.hour)
            return list(self.artifact.avail_index.get((week_type, weekday, hour), []))

        # Advanced Mode
        day = ts.date()
        # 1. Filter: Resource must be active on this specific date
        present_set = self.artifact.present_by_date.get(day, set())
        if not present_set:
            return []

        # 2. Filter: Resource must be active in this specific 2h bucket
        month = int(ts.month)
        weekday = int(ts.weekday())
        bucket2h = int(ts.hour // 2)

        candidates = self.artifact.monthly_index.get((month, weekday, bucket2h), [])
        if not candidates:
            return []

        # Intersection: Active in bucket AND Present today
        return [r for r in candidates if r in present_set]