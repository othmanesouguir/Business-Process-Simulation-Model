"""
src/resource_selector_1_7.py

Module 1.7 — Resource Selection Strategy.

This module implements the logic for choosing a specific resource from a list
of eligible candidates. While the current implementation uses a randomized
dispatching policy, this class is designed to be extensible for future strategies
(e.g., Least Busy, Shortest Queue, or Expertise-Based routing).

Key Features:
- Deterministic Randomness: Uses a fixed seed to ensure simulation reproducibility.
- Safety: Handles empty candidate lists gracefully.
"""

from __future__ import annotations

import random
from typing import List, Optional


class ResourceSelector:
    """
    Implements a Random Selection strategy for resource allocation.
    
    When multiple resources are available and qualified to perform a task,
    this selector chooses one uniformly at random. This baseline strategy
    distributes workload roughly evenly over time among available pools.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the selector with a random seed.
        
        Args:
            seed: Seed for the random number generator to ensure 
                  that simulation runs are reproducible.
        """
        self.rng = random.Random(seed)

    def select(self, candidates: List[str]) -> Optional[str]:
        """
        Selects a single resource from a list of candidates.

        Args:
            candidates: A list (or iterable) of resource IDs (strings) 
                        who are currently available and authorized.

        Returns:
            The ID of the selected resource, or None if the candidate list is empty.
        """
        if not candidates:
            return None
        
        # Convert to list to ensure indexability for random choice
        return self.rng.choice(list(candidates))