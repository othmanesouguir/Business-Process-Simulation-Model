"""
processing_time_predictor.py

Module 1.3 — Processing time predictor (runtime wrapper used by 1.1)

Training happens offline in:
    processing_times_TRAIN.py
which produces a cached artifact (pkl). This wrapper only loads it and serves queries.

Logic:
  - If activity starts with A_ or O_: return 0.0
  - Else:
      * if a BASIC distribution exists -> sample from it
      * else if an ADVANCED ML model exists -> predict via LightGBM quantile regression
      * else -> fallback (data-driven median)
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import warnings
from scipy import stats

warnings.filterwarnings("ignore")


class ProcessingTimePredictor:
    def __init__(self, model_path: str):
        # ----------------------------
        # Resolve model path into /models
        # ----------------------------
        p = Path(str(model_path))

        if not p.exists():
            # project root = parent of src/
            project_root = Path(__file__).resolve().parents[1]  # project/
            candidate = project_root / "models" / p.name
            if candidate.exists():
                p = candidate

        if not p.exists():
            raise FileNotFoundError(
                f"ProcessingTimePredictor: cannot find model artifact.\n"
                f"Given: {model_path}\n"
                f"Tried: {p}\n"
                f"Hint: put the .pkl in project/models/"
            )

        self.model_path = str(p)
        artifact = joblib.load(self.model_path)

        self.basic_distributions: Dict[str, dict] = artifact.get("basic_distributions", {})
        self.advanced_models: Dict[str, dict] = artifact.get("advanced_models", {})
        self.fallback_sec: float = float(artifact.get("fallback_sec", 600.0))

        # Used by 1.1 advanced/basic interface
        self.quantiles = [0.1, 0.5, 0.9]

    # ----------------------------
    # Basic distribution sampling
    # ----------------------------
    def _sample_basic(self, activity: str) -> float:
        info = self.basic_distributions.get(activity)
        if not info:
            return float(self.fallback_sec)

        dist = info.get("dist")
        params = info.get("params")

        if dist == "lognorm":
            shape, loc, scale = params
            return float(stats.lognorm.rvs(shape, loc=loc, scale=scale))
        if dist == "gamma":
            a, loc, scale = params
            return float(stats.gamma.rvs(a, loc=loc, scale=scale))
        if dist == "expon":
            loc, scale = params
            return float(stats.expon.rvs(loc=loc, scale=scale))
        if dist == "norm":
            mu, sigma = params
            return float(max(0.0, stats.norm.rvs(mu, sigma)))

        return float(self.fallback_sec)

    def _predict_basic(self, activity: str) -> float:
        info = self.basic_distributions.get(activity)
        if not info:
            return float(self.fallback_sec)
        return float(info.get("median", self.fallback_sec))

    # ----------------------------
    # Advanced ML features
    # ----------------------------
    def _build_features(
            self,
            activity: str,
            case_attributes: dict,
            resource: str,
            current_time: datetime,
            feature_cols: list,
            median_fill: dict,
            feature_name_map: dict | None = None,
    ) -> pd.DataFrame:
        row = {}

        # keep only what exists in feature_cols (no extra noise)
        for c in feature_cols:
            row[c] = None

        # ✅ apply mapping: original_name -> safe_name
        feature_name_map = feature_name_map or {}

        for k, v in (case_attributes or {}).items():
            kk = feature_name_map.get(k, k)  # map "case:LoanGoal" -> "case_LoanGoal"
            if kk in row:
                row[kk] = v

        # optional resource feature
        if "org_resource" in row:
            row["org_resource"] = resource
        elif "org:resource" in row:
            row["org:resource"] = resource

        df = pd.DataFrame([row])

        # categorical safety: convert objects to codes if needed
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

        # fill remaining NaNs
        for col, med in (median_fill or {}).items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        df = df.fillna(0)
        return df

    # ----------------------------
    # Public API used by 1.1
    # ----------------------------
    def sample_duration(
        self,
        activity: str,
        case_attributes: Optional[dict] = None,
        resource: str = "",
        current_time: Optional[datetime] = None,
        method: str = "median",
        quantile: float = 0.5,
        return_all_quantiles: bool = False,
    ):
        if current_time is None:
            current_time = datetime.utcnow()

        # A_ and O_ are treated as instant
        if activity.startswith("A_") or activity.startswith("O_"):
            if return_all_quantiles:
                return {q: 0.0 for q in self.quantiles}
            return 0.0

        # If we have a basic distribution -> use it
        if activity in self.basic_distributions:
            if method == "sample":
                v = float(self._sample_basic(activity))
            else:
                v = float(self._predict_basic(activity))

            v = max(0.0, v)
            if return_all_quantiles:
                return {q: v for q in self.quantiles}
            return v

        # Advanced ML path if available
        adv = self.advanced_models.get(activity)
        if not adv:
            if return_all_quantiles:
                return {q: float(self.fallback_sec) for q in self.quantiles}
            return float(self.fallback_sec)

        models = adv.get("models", {})
        feature_cols = adv.get("feature_cols", [])
        median_fill = adv.get("median_fill", {})

        if not models or not feature_cols:
            if return_all_quantiles:
                return {q: float(self.fallback_sec) for q in self.quantiles}
            return float(self.fallback_sec)

        feature_name_map = adv.get("feature_name_map", {})

        X = self._build_features(
            activity=activity,
            case_attributes=case_attributes or {},
            resource=resource,
            current_time=current_time,
            feature_cols=feature_cols,
            median_fill=median_fill,
            feature_name_map=feature_name_map,
        )

        if return_all_quantiles:
            out = {}
            for q in self.quantiles:
                model = models.get(q)
                if model is None:
                    out[q] = float(self.fallback_sec)
                else:
                    out[q] = float(model.predict(X)[0])
            return out

        # single quantile
        if quantile not in models:
            quantile = 0.5

        pred = float(models[quantile].predict(X)[0])
        return max(0.0, pred)
