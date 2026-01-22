"""
src/bpmn_adapter.py

BPMN Adapter Module.

This module parses a standard BPMN 2.0 XML file to extract the control-flow logic.
It provides an interface for the Simulation Engine to query:
1.  **Start Activity**: Where does a new case begin?
2.  **Next Activities**: Given current task A, what are valid next tasks B, C...?
3.  **Process End**: Is the current task a final state?

Logic:
- Parsing: Uses ElementTree to traverse standard BPMN tags (task, gateway, sequenceFlow).
- Graph Traversal: Uses BFS to skip over gateways and find the next *executable* task.
- XOR Support: Automatically returns multiple options for XOR gateways, allowing the 
  Next Activity Predictor (1.4) to choose the correct path based on data.

Usage:
    bpmn = BPMNAdapter("data/Signavio_Model.bpmn")
    next_tasks = bpmn.allowed_next("W_Validate application")
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Optional

# Standard BPMN Namespace
BPMN_NS = {"b": "http://www.omg.org/spec/BPMN/20100524/MODEL"}

# Tags representing executable work
TASK_TAGS = {
    "task", "userTask", "serviceTask", "manualTask", "businessRuleTask",
    "sendTask", "receiveTask", "scriptTask", "callActivity", "subProcess"
}

# Tags representing logic (skipped during task traversal)
GATEWAY_TAGS = {
    "exclusiveGateway", "parallelGateway", "inclusiveGateway", "eventBasedGateway"
}


class BPMNAdapter:
    """
    Parses a BPMN XML file and exposes the control flow graph.
    Designed to work seamlessly with the Discrete Event Simulator.
    """

    def __init__(self, bpmn_path: str):
        """
        Initialize the adapter by parsing the BPMN file.
        
        Args:
            bpmn_path: Path to the .bpmn file (absolute or relative).
        """
        # ----------------------------
        # Robust Path Resolution
        # ----------------------------
        p = Path(str(bpmn_path))

        # Check relative to project root/data if direct path fails
        if not p.exists():
            project_root = Path(__file__).resolve().parents[1]  # project/
            candidate = project_root / "data" / p.name
            if candidate.exists():
                p = candidate

        if not p.exists():
            raise FileNotFoundError(
                f"BPMNAdapter: Cannot find BPMN file.\n"
                f"Given: {bpmn_path}\n"
                f"Tried: {p}\n"
                f"Hint: Ensure .bpmn file exists in project/data/"
            )

        self.path = str(p)

        # Internal Graph Storage
        self.id_to_name: Dict[str, str] = {}
        self.id_to_kind: Dict[str, str] = {}  # 'task', 'gateway', 'start', 'end'
        self.name_to_id: Dict[str, str] = {}
        self.edges: Dict[str, List[str]] = {}
        
        self.start_id: Optional[str] = None
        self.end_ids: Set[str] = set()

        # Parse XML immediately
        self._parse()

        # Runtime Cache: activity_name -> list[next_activity_names]
        self._next_cache: Dict[str, List[str]] = {}

        # Pre-compute start activity
        self._start_activity = self._first_task_after(self.start_id) if self.start_id else None
        if self._start_activity is None:
            raise RuntimeError("Invalid BPMN: Could not find a reachable start activity.")

    def _parse(self):
        """Parses XML structure to build the internal graph representation."""
        tree = ET.parse(self.path)
        root = tree.getroot()

        # 1. Start Events
        start_events = root.findall(".//b:startEvent", BPMN_NS)
        if start_events:
            self.start_id = start_events[0].get("id")
            if self.start_id:
                self.id_to_kind[self.start_id] = "start"

        # 2. End Events
        for e in root.findall(".//b:endEvent", BPMN_NS):
            eid = e.get("id")
            if eid:
                self.end_ids.add(eid)
                self.id_to_kind[eid] = "end"

        # 3. Tasks
        for tag in TASK_TAGS:
            for t in root.findall(f".//b:{tag}", BPMN_NS):
                tid = t.get("id")
                name = t.get("name") or ""
                if not tid:
                    continue
                self.id_to_kind[tid] = "task"
                self.id_to_name[tid] = name
                # Map Name -> ID (First encounter wins if duplicate names exist)
                if name and name not in self.name_to_id:
                    self.name_to_id[name] = tid

        # 4. Gateways
        for tag in GATEWAY_TAGS:
            for g in root.findall(f".//b:{tag}", BPMN_NS):
                gid = g.get("id")
                if gid:
                    self.id_to_kind[gid] = "gateway"

        # 5. Sequence Flows (Edges)
        for sf in root.findall(".//b:sequenceFlow", BPMN_NS):
            src = sf.get("sourceRef")
            tgt = sf.get("targetRef")
            if src and tgt:
                self.edges.setdefault(src, []).append(tgt)

    def _is_task_id(self, node_id: str) -> bool:
        """Helper to check if a node ID corresponds to a Task."""
        return self.id_to_kind.get(node_id) == "task"

    def _is_end_id(self, node_id: str) -> bool:
        """Helper to check if a node ID corresponds to an End Event."""
        return self.id_to_kind.get(node_id) == "end"

    def _first_task_after(self, node_id: Optional[str]) -> Optional[str]:
        """
        Traverses graph via BFS starting from node_id.
        Stops and returns the name of the first 'Task' node encountered.
        Skips over Gateways.
        """
        if node_id is None:
            return None
            
        seen = set()
        q = [node_id]
        
        while q:
            cur = q.pop(0)
            if cur in seen:
                continue
            seen.add(cur)

            for nxt in self.edges.get(cur, []):
                # Found a task? Return its name.
                if self._is_task_id(nxt):
                    return self.id_to_name.get(nxt, "")
                
                # If not an end event, keep searching (it's likely a gateway)
                if not self._is_end_id(nxt):
                    q.append(nxt)
        return None

    # ----------------------------
    # Public API for Simulation Engine
    # ----------------------------
    def start_activity(self, case_attrs=None) -> str:
        """Returns the name of the activity that starts a new case."""
        return self._start_activity

    def allowed_next(self, current_activity: str, case_attrs=None) -> List[str]:
        """
        Returns a list of all valid next activities from the current state.
        Handles XOR gateways transparently by returning multiple options.
        """
        # Return cached result if available
        if current_activity in self._next_cache:
            return self._next_cache[current_activity]

        cur_id = self.name_to_id.get(current_activity)
        if not cur_id:
            # Case: Activity name not found in BPMN (should be rare)
            self._next_cache[current_activity] = []
            return []

        next_tasks: Set[str] = set()
        seen = set()
        q = [cur_id]

        # BFS to find reachable tasks, skipping intermediate gateways
        while q:
            node = q.pop(0)
            if node in seen:
                continue
            seen.add(node)

            for nxt in self.edges.get(node, []):
                if self._is_end_id(nxt):
                    continue
                
                if self._is_task_id(nxt):
                    nm = self.id_to_name.get(nxt, "")
                    if nm:
                        next_tasks.add(nm)
                else:
                    # Gateway or intermediate event -> continue traversing
                    q.append(nxt)

        out = sorted(next_tasks)
        self._next_cache[current_activity] = out
        return out

    def is_final(self, activity: str) -> bool:
        """
        Returns True if the activity leads to an End Event with no further tasks.
        """
        return len(self.allowed_next(activity)) == 0