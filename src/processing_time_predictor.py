"""
src/processing_time_predictor.py

Module 1.3 — Processing Time Predictor (Runtime Wrapper).

This module serves as the inference engine for activity durations. It loads
pre-trained models (produced by `processing_times_TRAIN.py`) and provides
time estimates based on activity type, resource, and case context.

Logic:
1. System Tasks (A_*, O_*): Return 0.0 (Instantaneous).
2. Basic Activities: Sample from fitted statistical distributions (LogNorm, Gamma, etc.).
3. Complex Activities: Predict using LightGBM Quantile Regression.
4. Fallback: Return global median duration.
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
    """
    Runtime wrapper for Duration Prediction.
    Loads a .pkl artifact containing both basic statistical distributions
    and advanced LightGBM models.
    """

    def __init__(self, model_path: str):
        """
        Initialize the predictor by loading the trained model artifact.
        
        Args:
            model_path: Path to the .pkl file (absolute or relative to project root).
        """
        # Resolve model path (handle relative paths robustly)
        p = Path(str(model_path))

        if not p.exists():
            # Check relative to project root (parent of src/)
            project_root = Path(__file__).resolve().parents[1]
            candidate = project_root / "models" / p.name
            if candidate.exists():
                p = candidate

        if not p.exists():
            raise FileNotFoundError(
                f"ProcessingTimePredictor: Cannot find model artifact.\n"
                f"Given: {model_path}\n"
                f"Tried: {p}\n"
                f"Hint: Ensure 'processing_model_advanced.pkl' is in project/models/"
            )

        self.model_path = str(p)
        artifact = joblib.load(self.model_path)

        # Unpack components from the artifact
        self.basic_distributions: Dict[str, dict] = artifact.get("basic_distributions", {})
        self.advanced_models: Dict[str, dict] = artifact.get("advanced_models", {})
        self.fallback_sec: float = float(artifact.get("fallback_sec", 600.0))

        # Quantiles used by the 1.1 engine (Optimistic, Realistic, Pessimistic)
        self.quantiles = [0.1, 0.5, 0.9]

    # ----------------------------
    # Basic Distribution Sampling
    # ----------------------------
    def _sample_basic(self, activity: str) -> float:
        """Samples from a fitted scipy statistical distribution."""
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
        """Returns the median value from the basic distribution (no sampling)."""
        info = self.basic_distributions.get(activity)
        if not info:
            return float(self.fallback_sec)
        return float(info.get("median", self.fallback_sec))

    # ----------------------------
    # Advanced ML Feature Engineering
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
        """
        Constructs a single-row DataFrame compatible with the trained LightGBM model.
        Applies column mapping and median filling for missing values.
        """
        row = {}

        # Initialize all expected columns
        for c in feature_cols:
            row[c] = None

        # Apply mapping: original_name -> safe_name (sanitized for LightGBM)
        feature_name_map = feature_name_map or {}

        for k, v in (case_attributes or {}).items():
            kk = feature_name_map.get(k, k)
            if kk in row:
                row[kk] = v

        # Add resource features if expected
        if "org_resource" in row:
            row["org_resource"] = resource
        elif "org:resource" in row:
            row["org:resource"] = resource

        df = pd.DataFrame([row])

        # Convert categorical objects to codes (LightGBM requirement)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

        # Fill missing values with training set medians
        for col, med in (median_fill or {}).items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        df = df.fillna(0)
        return df

    # ----------------------------
    # Public API (Used by Simulation Engine)
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
        """
        Main entry point to predict duration for an activity instance.
        
        Args:
            activity: The name of the task.
            case_attributes: Dictionary of case data (LoanAmount, etc.).
            resource: The resource assigned to the task.
            current_time: The simulation timestamp.
            method: 'median' (deterministic) or 'sample' (stochastic).
            quantile: The specific quantile to predict (if using ML).
            return_all_quantiles: If True, returns dict {0.1: val, 0.5: val, 0.9: val}.
        """
        if current_time is None:
            current_time = datetime.utcnow()

        # 1. System Tasks are instantaneous
        if activity.startswith("A_") or activity.startswith("O_"):
            if return_all_quantiles:
                return {q: 0.0 for q in self.quantiles}
            return 0.0

        # 2. Basic Distribution Strategy
        if activity in self.basic_distributions:
            if method == "sample":
                v = float(self._sample_basic(activity))
            else:
                v = float(self._predict_basic(activity))

            v = max(0.0, v)
            if return_all_quantiles:
                return {q: v for q in self.quantiles}
            return v

        # 3. Advanced ML Strategy
        adv = self.advanced_models.get(activity)
        if not adv:
            # No model found -> Fallback
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

        # Single quantile prediction
        if quantile not in models:
            quantile = 0.5

        pred = float(models[quantile].predict(X)[0])
        return max(0.0, pred)