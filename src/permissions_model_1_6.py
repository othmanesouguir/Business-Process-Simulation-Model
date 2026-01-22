"""
src/permissions_model_1_6.py

Module 1.6 — Resource Permissions Modeling.

This module determines which resources are authorized to execute which activities.
It offers two modes of operation:

1.  **Basic Mode**:
    - Direct Mapping: If a resource executed activity A in the logs, they are allowed to do it.
    
2.  **Advanced Mode (Implication Mining)**:
    - Builds an implication graph where Activity B -> Activity A if 
      resources who do B almost always do A (overlap >= 95%).
    - Collapses Strongly Connected Components (SCCs) to handle cyclic dependencies.
    - Expands permissions transitively (if User can do B, and B implies A, User can do A).
    
Usage:
    perms = PermissionsModel("data/bpi2017.csv", mode="advanced")
    can_do = perms.can_execute("User_1", "W_Validate application")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple
from collections import defaultdict

import joblib
import pandas as pd
import numpy as np


def _load_perm_df(csv_path: str) -> pd.DataFrame:
    """
    Loads necessary columns from the CSV and repairs any malformed rows
    (common in the BPI 2017 dataset where columns shift).
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Repair collapsed rows stored in "Action" column
    if {"Action", "concept:name", "org:resource"}.issubset(df.columns):
        broken = (
            (df["concept:name"].isna() | df["org:resource"].isna())
            & df["Action"].astype(str).str.contains(",", regex=False)
        )
        if broken.any():
            parts = df.loc[broken, "Action"].astype(str).str.split(",", n=18, expand=True)
            df.loc[broken & df["concept:name"].isna(), "concept:name"] = parts[0].values
            df.loc[broken & df["org:resource"].isna(), "org:resource"] = parts[1].values

    df = df[["concept:name", "org:resource"]].dropna().copy()
    df["concept:name"] = df["concept:name"].astype(str).str.strip()
    df["org:resource"] = df["org:resource"].astype(str).str.strip()

    # Filter for valid BPI 2017 activities (A_*, O_*, W_*)
    df = df[df["concept:name"].str.match(r"^[AOW]_", na=False)].copy()
    return df


@dataclass(frozen=True)
class PermissionArtifactBasic:
    """Artifact for Basic Mode: Simple Activity -> Resource List map."""
    activity_to_resources: Dict[str, List[str]]


@dataclass(frozen=True)
class PermissionArtifactAdvanced:
    """
    Artifact for Advanced Mode: 
    Includes inferred permissions derived from implication mining.
    """
    activity_to_resources_basic: Dict[str, List[str]]
    resource_base_perms: Dict[str, Set[str]]     # Direct history
    resource_expanded_perms: Dict[str, Set[str]] # Inferred history (Transitive Closure)


def _tarjan_scc(nodes: List[str], graph: Dict[str, Set[str]]) -> List[List[str]]:
    """
    Tarjan's Algorithm to find Strongly Connected Components (SCCs).
    Used to handle cycles in the permission implication graph.
    """
    index = 0
    stack: List[str] = []
    onstack: Set[str] = set()
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    sccs: List[List[str]] = []

    def strongconnect(v: str):
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1

        stack.append(v)
        onstack.add(v)

        for w in graph.get(v, set()):
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in nodes:
        if v not in idx:
            strongconnect(v)

    return sccs


def _train_basic(df_perm: pd.DataFrame) -> PermissionArtifactBasic:
    """Trains the Basic model (Direct execution history only)."""
    activity_to_resources = (
        df_perm.drop_duplicates()
        .groupby("concept:name", observed=True)["org:resource"]
        .apply(lambda s: s.astype(str).tolist())
        .to_dict()
    )
    return PermissionArtifactBasic(activity_to_resources=activity_to_resources)


def _train_advanced(
    df_perm: pd.DataFrame,
    *,
    min_support: int = 10,
    min_imp: float = 0.95,
) -> PermissionArtifactAdvanced:
    """
    Trains the Advanced model using Implication Mining.
    
    1. Base Permissions: Who actually did what?
    2. Implication Graph: If (User does B -> User does A), then B implies A.
    3. Graph Reduction: Collapse SCCs to DAG.
    4. Transitive Closure: Expand permissions based on DAG.
    """
    # 1. Base Mapping
    activity_to_resources_basic = (
        df_perm.drop_duplicates()
        .groupby("concept:name", observed=True)["org:resource"]
        .apply(lambda s: s.astype(str).tolist())
        .to_dict()
    )

    pairs = df_perm.drop_duplicates().copy()
    activity_sets: Dict[str, Set[str]] = (
        pairs.groupby("concept:name", observed=True)["org:resource"]
        .apply(lambda s: set(s.astype(str).tolist()))
        .to_dict()
    )

    activities = list(activity_sets.keys())

    # Build Base Resource Permissions
    resource_base: DefaultDict[str, Set[str]] = defaultdict(set)
    for act, res_set in activity_sets.items():
        for r in res_set:
            resource_base[r].add(act)

    # 2. Build Implication Graph
    imp_graph: Dict[str, Set[str]] = {a: set() for a in activities}

    for B in activities:
        RB = activity_sets[B]
        suppB = len(RB)
        if suppB < min_support:
            continue
        for A in activities:
            if A == B:
                continue
            # Overlap: How many users who do B also do A?
            inter = len(RB & activity_sets[A])
            imp = inter / suppB if suppB > 0 else 0.0
            if imp >= min_imp:
                imp_graph[B].add(A)

    # 3. Collapse SCCs (Handle cycles like A->B->A)
    sccs = _tarjan_scc(activities, imp_graph)
    activity_to_group: Dict[str, int] = {}
    for gid, group in enumerate(sccs):
        for act in group:
            activity_to_group[act] = gid

    group_to_activities: Dict[int, Set[str]] = {gid: set(group) for gid, group in enumerate(sccs)}

    # Build Group DAG
    group_edges: DefaultDict[int, Set[int]] = defaultdict(set)
    for B, outs in imp_graph.items():
        gB = activity_to_group[B]
        for A in outs:
            gA = activity_to_group[A]
            if gA != gB:
                group_edges[gB].add(gA)

    # 4. Compute Reachability (DFS on DAG)
    group_reach: Dict[int, Set[int]] = {}

    def dfs(g: int) -> Set[int]:
        if g in group_reach:
            return group_reach[g]
        reach = {g}
        for nxt in group_edges.get(g, set()):
            reach |= dfs(nxt)
        group_reach[g] = reach
        return reach

    for g in range(len(sccs)):
        dfs(g)

    # 5. Expand Permissions
    resource_expanded: Dict[str, Set[str]] = {}
    for r, base_acts in resource_base.items():
        # Get all groups the user has direct access to
        base_groups = {activity_to_group[a] for a in base_acts}
        expanded_groups: Set[int] = set()
        
        # Add all implied groups (Transitive)
        for g in base_groups:
            expanded_groups |= group_reach[g]
            
        # Convert groups back to specific activities
        expanded_acts: Set[str] = set()
        for g in expanded_groups:
            expanded_acts |= group_to_activities[g]
        resource_expanded[r] = expanded_acts

    return PermissionArtifactAdvanced(
        activity_to_resources_basic=activity_to_resources_basic,
        resource_base_perms=dict(resource_base),
        resource_expanded_perms=resource_expanded,
    )


class PermissionsModel:
    """
    Public Interface for Permissions Modeling (Task 1.6).
    Handles caching (loading/saving .pkl files) and runtime querying.
    """

    def __init__(
        self,
        csv_path: str,
        mode: str = "basic",
        *,
        cache_path: Optional[str] = None,
        min_support: int = 10,
        min_imp: float = 0.95,
    ):
        self.csv_path = str(csv_path)
        self.mode = mode.lower().strip()
        if self.mode not in {"basic", "advanced"}:
            raise ValueError("mode must be 'basic' or 'advanced'")

        self.min_support = int(min_support)
        self.min_imp = float(min_imp)

        # Default Cache Path: models/permissions_model_1_6_{mode}.pkl
        if cache_path is None:
            project_root = Path(__file__).resolve().parents[1]
            models_dir = project_root / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            cache_path = str(models_dir / f"permissions_model_1_6_{self.mode}.pkl")

        self.cache_path = cache_path
        self.artifact = self._load_or_train()

    def _load_or_train(self):
        """Loads cached model if available, otherwise triggers training."""
        p = Path(self.cache_path)
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                pass

        print(f"[PermissionsModel] Training new model (Mode: {self.mode})...")
        df_perm = _load_perm_df(self.csv_path)
        
        if self.mode == "basic":
            art = _train_basic(df_perm)
        else:
            art = _train_advanced(df_perm, min_support=self.min_support, min_imp=self.min_imp)

        try:
            joblib.dump(art, p)
            print(f"[PermissionsModel] Saved artifact to {p}")
        except Exception:
            pass

        return art

    def get_allowed_resources(self, activity: str) -> List[str]:
        """Returns list of all resources authorized to perform an activity."""
        activity = str(activity)
        if self.mode == "basic":
            return list(self.artifact.activity_to_resources.get(activity, []))

        # In Advanced mode, we reverse map the expanded permissions
        out: List[str] = []
        for r, perms in self.artifact.resource_expanded_perms.items():
            if activity in perms:
                out.append(r)
        return out

    def can_execute(self, resource: str, activity: str) -> bool:
        """Boolean check: Is this specific resource allowed to do this activity?"""
        resource = str(resource)
        activity = str(activity)
        
        if self.mode == "basic":
            return resource in set(self.artifact.activity_to_resources.get(activity, []))
            
        return activity in self.artifact.resource_expanded_perms.get(resource, set())