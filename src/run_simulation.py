"""
src/run_simulation.py

One-Click Simulation Pipeline.

This script acts as the main entry point for the simulation project.
It orchestrates the entire lifecycle:
1.  Checks if trained Machine Learning models exist in `models/`.
2.  If missing, it triggers training automatically (Module 1.3 & 1.4).
3.  Initializes all simulation components (Arrivals, Availability, etc.).
4.  Runs the Discrete Event Simulation for the specified timeframe (2016).
5.  Outputs the results to `outputs/` in both CSV and XES formats.

Folder Layout:
  project_root/
    data/          bpi2017.csv + Signavio_Model.bpmn
    src/           Module scripts (1.1 - 1.7)
    models/        Cached .pkl artifacts (created automatically)
    outputs/       simulated_log.csv, simulated_log.xes
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np

# --- imports from src/ modules ---
from arrival_model_1_2 import ArrivalProcess
from resource_availability_1_5 import ResourceAvailabilityModel
from permissions_model_1_6 import PermissionsModel
from resource_selector_1_7 import ResourceSelector
from processing_time_predictor import ProcessingTimePredictor
from simulation_engine_1_1 import SimulationEngine
from bpmn_adapter import BPMNAdapter

# --- import training entry points ---
from next_activity_TRAIN_1_4 import load_log as train14_load_log, train_next_activity_model
from processing_times_TRAIN import train_processing_model


# ----------------------------
# 1.4 Predictor (Runtime Wrapper)
# ----------------------------
class NextActivityPredictor:
    """
    Runtime wrapper for the Next Activity Prediction Model (Module 1.4).
    
    Loads the pre-trained Bigram model (next_activity_bigram_model.pkl).
    Logic:
      1. Try Bigram: P(next | prev2, prev1)
      2. Backoff to Unigram: P(next | prev1)
      3. Backoff to Global: P(next)
      
    Features:
      - Supports 'allowed_next' filter to respect BPMN XOR gateways.
      - Handles dead-ends gracefully.
    """

    def __init__(self, model_path: str, seed: int = 42):
        self.model = joblib.load(model_path)
        self.rng = np.random.default_rng(seed)

    def _sample_from(
        self,
        next_list: List[str],
        prob_array: np.ndarray,
        allowed_next: Optional[List[str]] = None,
    ) -> str:
        """Helper to sample a next activity respecting allowed transitions."""
        next_list = list(next_list)
        prob = np.array(prob_array, dtype=float)

        # No restrictions -> straightforward sampling
        if allowed_next is None or len(allowed_next) == 0:
            return str(self.rng.choice(next_list, p=prob))

        # Filter probabilities based on allowed_next (BPMN compliance)
        allowed_set = set(allowed_next)
        mask = np.array([n in allowed_set for n in next_list], dtype=bool)

        # Fallback: if model predicts nothing allowed, pick uniform random allowed
        if not mask.any():
            return str(self.rng.choice(list(allowed_set)))

        # Re-normalize probabilities for allowed transitions
        next_f = [n for n, m in zip(next_list, mask) if m]
        prob_f = prob[mask]
        s = float(prob_f.sum())

        if not np.isfinite(s) or s <= 0:
            return str(self.rng.choice(next_f))

        prob_f = prob_f / s
        return str(self.rng.choice(next_f, p=prob_f))

    def sample_next(
        self,
        prev2: Optional[str],
        prev1: Optional[str],
        allowed_next: Optional[List[str]] = None,
    ) -> str:
        """Determines the next activity in the process trace."""
        # Level 1: Bigram Context
        if prev2 is not None and prev1 is not None:
            key = (prev2, prev1)
            if key in self.model.get("bigram", {}):
                info = self.model["bigram"][key]
                return self._sample_from(info["next"], info["prob"], allowed_next)

        # Level 2: Unigram Context
        if prev1 is not None and prev1 in self.model.get("unigram", {}):
            info = self.model["unigram"][prev1]
            return self._sample_from(info["next"], info["prob"], allowed_next)

        # Level 3: Global Distribution
        info = self.model["global"]
        return self._sample_from(info["next"], info["prob"], allowed_next)


# ----------------------------
# Training Orchestrator
# ----------------------------
def train_if_missing(project_root: Path, csv_path: Path) -> None:
    """
    Checks for existence of trained models. Triggers training only if needed.
    """
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_13 = models_dir / "processing_model_advanced.pkl"
    model_14 = models_dir / "next_activity_bigram_model.pkl"

    # ---- 1.4 Next Activity Training ----
    if not model_14.exists():
        print("\n[TRAIN] 1.4 (Next Activity) model missing -> training now...")

        df = train14_load_log(xes_path=None, csv_path=str(csv_path))
        model, c_bigram, c_uni, c_glob = train_next_activity_model(df)

        joblib.dump(model, model_14)
        print(f"✅ Saved 1.4 model -> {model_14}")

    else:
        print("\n[SKIP] 1.4 model exists -> using cached pickle")

    # ---- 1.3 Processing Time Training ----
    if not model_13.exists():
        print("\n[TRAIN] 1.3 (Processing Time) model missing -> training now...")

        train_processing_model(
            log_path=str(csv_path),
            model_path=str(model_13),
        )

        if not model_13.exists():
            raise FileNotFoundError("1.3 training finished but output file not found.")

        print(f"✅ Saved 1.3 model -> {model_13}")

    else:
        print("\n[SKIP] 1.3 model exists -> using cached pickle")


# ----------------------------
# Main Execution
# ----------------------------
def main():
    # Setup Paths
    project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / "data"
    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "bpi2017.csv"
    bpmn_path = data_dir / "Signavio_Model.bpmn"

    # Validation
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Data File: {csv_path}")
    if not bpmn_path.exists():
        raise FileNotFoundError(f"Missing BPMN Model: {bpmn_path}")

    # 1. Train models (if needed)
    train_if_missing(project_root, csv_path)

    model_13_path = str(models_dir / "processing_model_advanced.pkl")
    model_14_path = str(models_dir / "next_activity_bigram_model.pkl")

    # 2. Configure Simulation
    mode = "advanced"  # Use 'advanced' for 2h buckets and ML prediction
    start_time = "2016-01-01 00:00:00+00:00"
    end_time = "2017-01-01 00:00:00+00:00"  # Full Year 2016

    out_csv = str(outputs_dir / f"simulated_log_{mode}_2016.csv")
    out_xes = str(outputs_dir / f"simulated_log_{mode}_2016.xes")

    # 3. Initialize Modules with Caching
    # Using cached artifacts speeds up initialization significantly
    arrivals_cache = str(models_dir / "arrival_model_1_2.pkl")
    availability_cache = str(models_dir / f"availability_{mode}_1_5.pkl")
    permissions_cache = str(models_dir / f"permissions_{mode}_1_6.pkl")

    arrivals = ArrivalProcess(csv_path=str(csv_path), seed=42, cache_path=arrivals_cache)
    
    availability = ResourceAvailabilityModel(
        csv_path=str(csv_path),
        mode=mode,
        cache_path=availability_cache,
    )
    
    permissions = PermissionsModel(
        csv_path=str(csv_path),
        mode=mode,
        cache_path=permissions_cache,
    )
    
    selector = ResourceSelector(seed=42)
    next_act = NextActivityPredictor(model_path=model_14_path, seed=42)
    duration = ProcessingTimePredictor(model_path=model_13_path)
    bpmn = BPMNAdapter(str(bpmn_path))

    # 4. Instantiate Engine
    print("\n[INFO] Initializing Simulation Engine...")
    engine = SimulationEngine(
        bpmn=bpmn,
        arrival_process=arrivals,
        duration_model=duration,
        next_activity_model=next_act,
        availability_model=availability,
        permissions_model=permissions,
        selector=selector,
        mode=mode,
        start_time=start_time,
        end_time=end_time,
        out_csv_path=out_csv,
        out_xes_path=out_xes, 
        seed=42,
    )

    # 5. Run
    print(f"[INFO] Running Simulation from {start_time} to {end_time}...")
    out = engine.run(max_cases=None)
    
    print("\n" + "="*40)
    print("✅ SIMULATION COMPLETE")
    print(f"📄 CSV Log: {out}")
    print(f"📄 XES Log: {out_xes}")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()