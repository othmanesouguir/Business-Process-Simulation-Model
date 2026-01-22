"""resource_selector_1_7.py

Module 1.7 — Resource selection.

For your assignment's simple version:
  - take list of candidate resources
  - return one randomly

API used by 1.1:
  selector.select(candidates) -> resource or None
"""

from __future__ import annotations

import random
from typing import List, Optional


class ResourceSelector:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def select(self, candidates: List[str]) -> Optional[str]:
        if not candidates:
            return None
        return self.rng.choice(list(candidates))
