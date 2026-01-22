"""
next_activity_predictor_1_4.py

Tiny wrapper to load the NextActivityPredictor from your peer's `1.4.basic.py`.

Why a wrapper?
--------------
Python can't import a module that starts with a number + has dots nicely.
So this loads it by file path.

Usage:
    from next_activity_predictor_1_4 import load_next_activity_predictor

    next_act = load_next_activity_predictor(
        model_path="models/next_activity_model_1_4_bigram.pkl",
        peer_py_path="1.4.basic.py"
    )
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def load_next_activity_predictor(
    model_path: str,
    peer_py_path: str = "1.4.basic.py",
    seed: int = 42
) -> Any:
    # --------------------------
    # Resolve peer (1.4.basic.py)
    # --------------------------
    peer_path = Path(peer_py_path)

    if not peer_path.exists():
        # ✅ try relative to THIS file (src/)
        peer_path = Path(__file__).resolve().parent / peer_py_path

    if not peer_path.exists():
        # ✅ try project root (project/)
        peer_path = Path(__file__).resolve().parents[1] / peer_py_path

    if not peer_path.exists():
        raise FileNotFoundError(f"Cannot find peer 1.4 file: {peer_py_path}")

    # --------------------------
    # Resolve model path
    # --------------------------
    mpath = Path(model_path)

    if not mpath.exists():
        # ✅ try models/ folder in project root
        project_root = Path(__file__).resolve().parents[1]
        candidate = project_root / "models" / model_path
        if candidate.exists():
            mpath = candidate

    if not mpath.exists():
        raise FileNotFoundError(f"Cannot find next-activity model file: {model_path}")

    # --------------------------
    # Load module by filepath
    # --------------------------
    spec = importlib.util.spec_from_file_location("peer_next_activity", str(peer_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {peer_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    if not hasattr(module, "NextActivityPredictor"):
        raise AttributeError("Peer 1.4 file does not define NextActivityPredictor")

    return module.NextActivityPredictor(str(mpath), seed=seed)
