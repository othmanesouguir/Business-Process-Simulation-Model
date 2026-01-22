# bpmn_adapter.py
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Optional

BPMN_NS = {"b": "http://www.omg.org/spec/BPMN/20100524/MODEL"}

TASK_TAGS = {
    "task", "userTask", "serviceTask", "manualTask", "businessRuleTask",
    "sendTask", "receiveTask", "scriptTask", "callActivity", "subProcess"
}
GATEWAY_TAGS = {"exclusiveGateway", "parallelGateway", "inclusiveGateway", "eventBasedGateway"}


class BPMNAdapter:
    """
    Minimal BPMN adapter for 1.1 engine.
    It supports XOR gateways automatically because allowed_next() returns >1 next tasks.
    """

    def __init__(self, bpmn_path: str):
        # ----------------------------
        # Resolve BPMN path (project-safe)
        # ----------------------------
        p = Path(str(bpmn_path))

        # If relative path doesn't exist, try project/data/<filename>
        if not p.exists():
            project_root = Path(__file__).resolve().parents[1]  # project/
            candidate = project_root / "data" / p.name
            if candidate.exists():
                p = candidate

        if not p.exists():
            raise FileNotFoundError(
                f"BPMNAdapter: cannot find BPMN file.\n"
                f"Given: {bpmn_path}\n"
                f"Tried: {p}\n"
                f"Hint: BPMN should be located in project/data/ (example: data/Signavio_Model.bpmn)"
            )

        self.path = str(p)

        self.id_to_name: Dict[str, str] = {}
        self.id_to_kind: Dict[str, str] = {}  # task/gateway/start/end/other
        self.name_to_id: Dict[str, str] = {}
        self.edges: Dict[str, List[str]] = {}
        self.start_id: Optional[str] = None
        self.end_ids: Set[str] = set()

        self._parse()

        # Cache: activity_name -> list[next_activity_names]
        self._next_cache: Dict[str, List[str]] = {}

        # compute first start activity once
        self._start_activity = self._first_task_after(self.start_id) if self.start_id else None
        if self._start_activity is None:
            raise RuntimeError("Could not find start activity in BPMN")

    def _parse(self):
        tree = ET.parse(self.path)
        root = tree.getroot()

        # Find start events
        start_events = root.findall(".//b:startEvent", BPMN_NS)
        if start_events:
            self.start_id = start_events[0].get("id")
            if self.start_id:
                self.id_to_kind[self.start_id] = "start"

        # Find end events
        for e in root.findall(".//b:endEvent", BPMN_NS):
            eid = e.get("id")
            if eid:
                self.end_ids.add(eid)
                self.id_to_kind[eid] = "end"

        # Find tasks
        for tag in TASK_TAGS:
            for t in root.findall(f".//b:{tag}", BPMN_NS):
                tid = t.get("id")
                name = t.get("name") or ""
                if not tid:
                    continue
                self.id_to_kind[tid] = "task"
                self.id_to_name[tid] = name
                if name and name not in self.name_to_id:
                    self.name_to_id[name] = tid

        # Find gateways
        for tag in GATEWAY_TAGS:
            for g in root.findall(f".//b:{tag}", BPMN_NS):
                gid = g.get("id")
                if gid:
                    self.id_to_kind[gid] = "gateway"

        # Build edges via sequenceFlow
        for sf in root.findall(".//b:sequenceFlow", BPMN_NS):
            src = sf.get("sourceRef")
            tgt = sf.get("targetRef")
            if src and tgt:
                self.edges.setdefault(src, []).append(tgt)

    def _is_task_id(self, node_id: str) -> bool:
        return self.id_to_kind.get(node_id) == "task"

    def _is_end_id(self, node_id: str) -> bool:
        return self.id_to_kind.get(node_id) == "end"

    def _first_task_after(self, node_id: Optional[str]) -> Optional[str]:
        """BFS from node_id until first task is found -> return its task name."""
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
                if self._is_task_id(nxt):
                    return self.id_to_name.get(nxt, "")
                if not self._is_end_id(nxt):
                    q.append(nxt)
        return None

    def start_activity(self, case_attrs=None) -> str:
        return self._start_activity

    def allowed_next(self, current_activity: str, case_attrs=None) -> List[str]:
        """Return all next tasks reachable without passing another task first."""
        if current_activity in self._next_cache:
            return self._next_cache[current_activity]

        cur_id = self.name_to_id.get(current_activity)
        if not cur_id:
            self._next_cache[current_activity] = []
            return []

        next_tasks: Set[str] = set()
        seen = set()
        q = [cur_id]

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
                    q.append(nxt)

        out = sorted(next_tasks)
        self._next_cache[current_activity] = out
        return out

    def is_final(self, activity: str) -> bool:
        """Final if no allowed next tasks."""
        return len(self.allowed_next(activity)) == 0
