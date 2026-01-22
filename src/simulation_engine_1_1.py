"""
simulation_engine_1_1.py

Module 1.1 — Discrete Event Simulation Engine (your exact design)

✅ FIXES:
- Still simulates arrivals, but DOES NOT log "case_arrival"
- DOES NOT log "case_end"
- Can export BOTH CSV and XES into outputs/

Uses:
  - 1.2 arrivals            -> arrival_process.next_arrival_time(t)
  - 1.3 duration prediction -> duration_model.sample_duration(...)
  - 1.4 next activity       -> next_activity_model.sample_next(prev2, prev1, allowed_next=[...])
  - 1.5 availability        -> availability_model.get_available_resources(t)
  - 1.6 permissions         -> permissions_model.can_execute(resource, activity)
  - 1.7 resource selection  -> selector.select(candidates)
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from collections import deque

import numpy as np
import pandas as pd

# ----------------------------
# Optional XES export
# ----------------------------
try:
    import pm4py
except Exception:
    pm4py = None


# ----------------------------
# Event types + ordering
# ----------------------------
class EventType:
    ACTIVITY_COMPLETE = "ACTIVITY_COMPLETE"
    RESOURCE_CHECK = "RESOURCE_CHECK"
    CASE_ARRIVAL = "CASE_ARRIVAL"
    RECHECK = "RECHECK"


# tie-break priority if same timestamp
# COMPLETE first -> RESOURCE_CHECK -> ARRIVAL -> RECHECK
_EVENT_PRIORITY = {
    EventType.ACTIVITY_COMPLETE: 0,
    EventType.RESOURCE_CHECK: 1,
    EventType.CASE_ARRIVAL: 2,
    EventType.RECHECK: 3,
}


@dataclass
class Event:
    ts: pd.Timestamp
    type: str
    payload: dict


@dataclass
class Task:
    case_id: str
    activity: str
    ready_ts: pd.Timestamp
    remaining_s: Optional[float] = None  # only for suspended tasks


@dataclass
class Execution:
    task_id: str
    case_id: str
    activity: str
    resource: str
    slice_start: pd.Timestamp
    remaining_s: float
    complete_ts: Optional[pd.Timestamp] = None  # only if guaranteed inside bucket


# ----------------------------
# Case attributes sampling (logical, data-driven)
# ----------------------------
class CaseAttributeSampler:
    """
    Samples realistic case attributes from historical BPIC CSV.

    - LoanGoal sampled by its real frequency
    - ApplicationType sampled by its real frequency
    - RequestedAmount sampled logically:
        prefer conditional by (LoanGoal, ApplicationType),
        then by LoanGoal,
        else global.
      + bounded by per-goal q05..q95 range
    """

    def __init__(self, csv_path: str, seed: int = 42):
        self.csv_path = str(csv_path)
        self.rng = np.random.default_rng(seed)

        self.loan_goals: List[str] = []
        self.loan_goal_p: np.ndarray = np.array([], dtype=float)

        self.app_types: List[str] = []
        self.app_type_p: np.ndarray = np.array([], dtype=float)

        self.amounts_global: np.ndarray = np.array([], dtype=float)
        self.global_lo: float = 1000.0
        self.global_hi: float = 50000.0

        self.amounts_by_goal: Dict[str, np.ndarray] = {}
        self.bounds_by_goal: Dict[str, Tuple[float, float]] = {}

        self.amounts_by_goal_app: Dict[Tuple[str, str], np.ndarray] = {}
        self.bounds_by_goal_app: Dict[Tuple[str, str], Tuple[float, float]] = {}

        self._load()

    def _safe_probs(self, counts: pd.Series) -> np.ndarray:
        p = counts.to_numpy(dtype=float)
        s = float(p.sum())
        if not np.isfinite(s) or s <= 0:
            return np.ones(len(p), dtype=float) / max(1, len(p))
        return p / s

    def _clean_str(self, s: pd.Series) -> pd.Series:
        s = s.dropna().astype(str).str.strip()
        return s[s != ""]

    def _compute_bounds(self, x: np.ndarray, lo_q=0.05, hi_q=0.95) -> Tuple[float, float]:
        x = x[np.isfinite(x)]
        x = x[x > 0]
        if x.size == 0:
            return (self.global_lo, self.global_hi)

        lo = float(np.quantile(x, lo_q))
        hi = float(np.quantile(x, hi_q))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return (self.global_lo, self.global_hi)

        return (lo, hi)

    def _load(self):
        try:
            df = pd.read_csv(self.csv_path, low_memory=False)
        except Exception:
            return

        # LoanGoal distribution
        if "case:LoanGoal" in df.columns:
            col = self._clean_str(df["case:LoanGoal"])
            if len(col) > 0:
                counts = col.value_counts()
                self.loan_goals = counts.index.tolist()
                self.loan_goal_p = self._safe_probs(counts)

        # ApplicationType distribution
        if "case:ApplicationType" in df.columns:
            col = self._clean_str(df["case:ApplicationType"])
            if len(col) > 0:
                counts = col.value_counts()
                self.app_types = counts.index.tolist()
                self.app_type_p = self._safe_probs(counts)

        # RequestedAmount global pool
        if "case:RequestedAmount" in df.columns:
            amt = pd.to_numeric(df["case:RequestedAmount"], errors="coerce").dropna()
            amt = amt[(amt > 0) & np.isfinite(amt)]
            if len(amt) > 0:
                self.amounts_global = amt.to_numpy(dtype=float)
                self.global_lo, self.global_hi = self._compute_bounds(self.amounts_global, 0.01, 0.99)

        # conditional pools
        if "case:RequestedAmount" in df.columns and self.amounts_global.size > 0:
            tmp = df.copy()
            tmp["case:RequestedAmount"] = pd.to_numeric(tmp["case:RequestedAmount"], errors="coerce")
            tmp = tmp.dropna(subset=["case:RequestedAmount"])
            tmp = tmp[(tmp["case:RequestedAmount"] > 0) & np.isfinite(tmp["case:RequestedAmount"])]

            has_goal = "case:LoanGoal" in tmp.columns
            has_app = "case:ApplicationType" in tmp.columns

            if has_goal:
                tmp["case:LoanGoal"] = tmp["case:LoanGoal"].astype(str).str.strip()
                for g, gg in tmp.groupby("case:LoanGoal"):
                    x = gg["case:RequestedAmount"].to_numpy(dtype=float)
                    if len(x) >= 80:
                        self.amounts_by_goal[g] = x
                        self.bounds_by_goal[g] = self._compute_bounds(x, 0.05, 0.95)

            if has_goal and has_app:
                tmp["case:ApplicationType"] = tmp["case:ApplicationType"].astype(str).str.strip()
                for (g, a), gg in tmp.groupby(["case:LoanGoal", "case:ApplicationType"]):
                    x = gg["case:RequestedAmount"].to_numpy(dtype=float)
                    if len(x) >= 80:
                        self.amounts_by_goal_app[(g, a)] = x
                        self.bounds_by_goal_app[(g, a)] = self._compute_bounds(x, 0.05, 0.95)

    def _round_amount(self, x: float) -> float:
        return float(int(round(x / 100.0)) * 100)

    def _sample_amount_logical(self, loan_goal: str, app_type: str) -> float:
        pool = self.amounts_by_goal_app.get((loan_goal, app_type))
        bounds = self.bounds_by_goal_app.get((loan_goal, app_type))

        if pool is None or bounds is None:
            pool = self.amounts_by_goal.get(loan_goal)
            bounds = self.bounds_by_goal.get(loan_goal)

        if pool is None or bounds is None:
            pool = self.amounts_global
            bounds = (self.global_lo, self.global_hi)

        lo, hi = bounds

        for _ in range(50):
            x = float(self.rng.choice(pool))
            if lo <= x <= hi:
                return self._round_amount(x)

        if self.amounts_global.size > 0:
            x = float(self.rng.choice(self.amounts_global))
        else:
            x = float(self.rng.integers(int(self.global_lo), int(self.global_hi)))

        x = float(np.clip(x, lo, hi))
        return self._round_amount(x)

    def sample(self) -> Dict[str, Any]:
        if self.loan_goals:
            loan_goal = str(self.rng.choice(self.loan_goals, p=self.loan_goal_p))
        else:
            loan_goal = "Unknown"

        if self.app_types:
            app_type = str(self.rng.choice(self.app_types, p=self.app_type_p))
        else:
            app_type = "Unknown"

        requested_amount = self._sample_amount_logical(loan_goal, app_type)

        out: Dict[str, Any] = {
            "case:LoanGoal": loan_goal,
            "case:ApplicationType": app_type,
            "case:RequestedAmount": requested_amount,
        }

        # extra safe numeric fields
        out.setdefault("CreditScore", int(self.rng.integers(300, 850)))
        out.setdefault("OfferedAmount", float(out["case:RequestedAmount"]))
        out.setdefault("NumberOfTerms", int(self.rng.choice([12, 24, 36, 48, 60])))
        out.setdefault("MonthlyCost", float(self.rng.integers(100, 1000)))
        out.setdefault("FirstWithdrawalAmount", float(self.rng.integers(500, 10000)))

        return out


# ----------------------------
# 1.1 Engine
# ----------------------------
class SimulationEngine:
    def __init__(
        self,
        *,
        bpmn: Any,
        arrival_process: Any,
        duration_model: Any,
        next_activity_model: Any,
        availability_model: Any,
        permissions_model: Any,
        selector: Any,
        mode: str = "basic",
        start_time: str | pd.Timestamp = "2016-01-01 00:00:00+00:00",
        end_time: str | pd.Timestamp = "2027-01-01 00:00:00+00:00",
        out_csv_path: str = "simulated_log.csv",
        out_xes_path: Optional[str] = None,
        seed: int = 42,
    ):
        self.mode = mode.lower().strip()
        assert self.mode in {"basic", "advanced"}

        self.bpmn = bpmn
        self.arrivals = arrival_process
        self.duration = duration_model
        self.next_act = next_activity_model
        self.availability = availability_model
        self.permissions = permissions_model
        self.selector = selector

        self.start_time = pd.Timestamp(start_time).tz_convert("UTC")
        self.end_time = pd.Timestamp(end_time).tz_convert("UTC")

        # ✅ Force output into outputs/ folder
        project_root = Path(__file__).resolve().parents[1]
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        out_csv_p = Path(str(out_csv_path))
        if not out_csv_p.is_absolute():
            out_csv_p = outputs_dir / out_csv_p.name
        self.out_csv_path = str(out_csv_p)

        self.out_xes_path = None
        if out_xes_path:
            out_xes_p = Path(str(out_xes_path))
            if not out_xes_p.is_absolute():
                out_xes_p = outputs_dir / out_xes_p.name
            self.out_xes_path = str(out_xes_p)

        self.rng = np.random.default_rng(seed)

        # EQ
        self.eventq: List[Tuple[pd.Timestamp, int, int, Event]] = []
        self.seq = itertools.count()

        # EP
        self.executing: Dict[str, Execution] = {}
        self.busy_resources: Set[str] = set()

        # SA / WA
        self.suspended: Deque[Task] = deque()
        self.waiting: Deque[Task] = deque()

        self.case_counter = 0
        self.task_counter = 0

        self.case_attrs: Dict[str, Dict[str, Any]] = {}
        self.case_last2: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        csv_path = getattr(arrival_process, "csv_path", "bpi2017.csv")
        self.attr_sampler = CaseAttributeSampler(csv_path, seed=seed)

        self.log_rows: List[Dict[str, Any]] = []

        self.bucket_hours = 1 if self.mode == "basic" else 2

    # ---------- EQ ----------
    def _push_event(self, ts: pd.Timestamp, etype: str, payload: Optional[dict] = None):
        ts = pd.Timestamp(ts).tz_convert("UTC")
        ev = Event(ts=ts, type=etype, payload=payload or {})
        prio = _EVENT_PRIORITY.get(etype, 99)
        heapq.heappush(self.eventq, (ts, prio, next(self.seq), ev))

    def _pop_event(self) -> Optional[Event]:
        if not self.eventq:
            return None
        return heapq.heappop(self.eventq)[-1]

    # ---------- Logging ----------
    def _log(self, ts: pd.Timestamp, case_id: str, activity: str, transition: str, resource: Optional[str]):
        self.log_rows.append(
            {
                "time:timestamp": pd.Timestamp(ts).tz_convert("UTC").isoformat(),
                "case:concept:name": case_id,
                "concept:name": activity,
                "lifecycle:transition": transition,
                "org:resource": resource if resource else "",
            }
        )

    def _enqueue_waiting(self, t: pd.Timestamp, task: Task):
        t = pd.Timestamp(t).tz_convert("UTC")
        self.waiting.append(task)
        # BPIC-style schedule log
        self._log(t, task.case_id, task.activity, "schedule", "")

    # ---------- Buckets ----------
    def _next_bucket_boundary(self, t: pd.Timestamp) -> pd.Timestamp:
        t = pd.Timestamp(t).tz_convert("UTC")
        t0 = t.floor("h")

        if (t - t0).total_seconds() == 0:
            return t0 + pd.Timedelta(hours=self.bucket_hours)

        nxt = t0 + pd.Timedelta(hours=1)

        if self.bucket_hours == 2:
            if int(nxt.hour) % 2 == 1:
                nxt = nxt + pd.Timedelta(hours=1)

        return nxt

    def _bucket_end(self, t: pd.Timestamp) -> pd.Timestamp:
        return self._next_bucket_boundary(t)

    # ---------- IDs ----------
    def _new_case_id(self) -> str:
        self.case_counter += 1
        return f"Case_{self.case_counter}"

    def _new_task_id(self) -> str:
        self.task_counter += 1
        return f"T{self.task_counter}"

    # ---------- Start/Resume ----------
    def _start_or_resume_task(self, t: pd.Timestamp, task: Task, resource: str, *, resumed: bool):
        t = pd.Timestamp(t).tz_convert("UTC")
        bucket_end = self._bucket_end(t)
        bucket_remaining_s = max(0.0, float((bucket_end - t).total_seconds()))

        if resumed and task.remaining_s is not None:
            dur_s = float(task.remaining_s)
        else:
            attrs = self.case_attrs[task.case_id]
            if hasattr(self.duration, "sample_duration"):
                dur_s = float(
                    self.duration.sample_duration(
                        activity=task.activity,
                        case_attributes=attrs,
                        resource=resource,
                        current_time=t.to_pydatetime(),
                        method="sample" if self.mode == "advanced" else "median",
                    )
                )
            elif hasattr(self.duration, "predict_duration"):
                dur_s = float(
                    self.duration.predict_duration(
                        activity=task.activity,
                        case_attributes=attrs,
                        resource=resource,
                        current_time=t.to_pydatetime(),
                        quantile=0.5,
                    )
                )
            else:
                dur_s = 60.0

        dur_s = max(1.0, float(dur_s))

        task_id = self._new_task_id()
        ex = Execution(
            task_id=task_id,
            case_id=task.case_id,
            activity=task.activity,
            resource=resource,
            slice_start=t,
            remaining_s=dur_s,
            complete_ts=None,
        )

        self.executing[task_id] = ex
        self.busy_resources.add(resource)

        self._log(t, task.case_id, task.activity, "resume" if resumed else "start", resource)

        # schedule completion ONLY if guaranteed inside bucket
        if dur_s <= bucket_remaining_s + 1e-9:
            complete_ts = t + pd.to_timedelta(dur_s, unit="s")
            ex.complete_ts = complete_ts
            self._push_event(complete_ts, EventType.ACTIVITY_COMPLETE, {"task_id": task_id})

    # ---------- Complete ----------
    def _complete_task(self, t: pd.Timestamp, task_id: str):
        t = pd.Timestamp(t).tz_convert("UTC")
        ex = self.executing.get(task_id)
        if ex is None:
            return

        if ex.complete_ts is None:
            return
        if abs((ex.complete_ts - t).total_seconds()) > 1e-6:
            return

        del self.executing[task_id]
        self.busy_resources.discard(ex.resource)

        self._log(t, ex.case_id, ex.activity, "complete", ex.resource)

        prev2, prev1 = self.case_last2.get(ex.case_id, (None, None))
        self.case_last2[ex.case_id] = (prev1, ex.activity)

        # ✅ FIX: do NOT log case_end
        if hasattr(self.bpmn, "is_final") and self.bpmn.is_final(ex.activity):
            return

        allowed_next = []
        if hasattr(self.bpmn, "allowed_next"):
            allowed_next = list(self.bpmn.allowed_next(ex.activity, self.case_attrs[ex.case_id]) or [])

        # ✅ FIX: do NOT log case_end
        if not allowed_next:
            return

        if len(allowed_next) == 1:
            nxt = allowed_next[0]
        else:
            prev2, prev1 = self.case_last2.get(ex.case_id, (None, None))
            if hasattr(self.next_act, "sample_next"):
                nxt = self.next_act.sample_next(prev2, prev1, allowed_next=allowed_next)
            else:
                nxt = str(self.rng.choice(allowed_next))

        self._enqueue_waiting(t, Task(case_id=ex.case_id, activity=nxt, ready_ts=t))
        self._push_event(t, EventType.RECHECK, {})

    # ---------- RESOURCE_CHECK ----------
    def _resource_check(self, t: pd.Timestamp):
        t = pd.Timestamp(t).tz_convert("UTC")

        available_now = set(self.availability.get_available_resources(t))
        bucket_end = self._bucket_end(t)
        bucket_remaining_s = max(0.0, float((bucket_end - t).total_seconds()))

        to_suspend: List[str] = []
        to_schedule_complete: List[str] = []

        for task_id, ex in list(self.executing.items()):
            elapsed = max(0.0, float((t - ex.slice_start).total_seconds()))
            ex.remaining_s = max(0.0, ex.remaining_s - elapsed)
            ex.slice_start = t

            # suspend at boundary if resource gone
            if ex.resource not in available_now:
                to_suspend.append(task_id)
                continue

            # schedule completion if now guaranteed in bucket
            if ex.complete_ts is None and ex.remaining_s <= bucket_remaining_s + 1e-9:
                to_schedule_complete.append(task_id)

        # handle suspensions
        for task_id in to_suspend:
            ex = self.executing.pop(task_id, None)
            if ex is None:
                continue

            self.busy_resources.discard(ex.resource)
            self._log(t, ex.case_id, ex.activity, "suspend", ex.resource)

            self.suspended.append(
                Task(
                    case_id=ex.case_id,
                    activity=ex.activity,
                    ready_ts=t,
                    remaining_s=float(ex.remaining_s),
                )
            )

        # handle newly schedulable completes
        for task_id in to_schedule_complete:
            ex = self.executing.get(task_id)
            if ex is None:
                continue
            complete_ts = t + pd.to_timedelta(ex.remaining_s, unit="s")
            ex.complete_ts = complete_ts
            self._push_event(complete_ts, EventType.ACTIVITY_COMPLETE, {"task_id": task_id})

        # dispatch same timestamp
        self._push_event(t, EventType.RECHECK, {})

        # schedule next bucket boundary
        nxt_boundary = t + pd.Timedelta(hours=self.bucket_hours)
        if nxt_boundary <= self.end_time:
            self._push_event(nxt_boundary, EventType.RESOURCE_CHECK, {})

    # ---------- RECHECK ----------
    def _allocate(self, t: pd.Timestamp):
        t = pd.Timestamp(t).tz_convert("UTC")

        while True:
            available = set(self.availability.get_available_resources(t))
            free_resources = list(available - self.busy_resources)
            if not free_resources:
                return

            def try_queue(queue: Deque[Task], resumed: bool) -> bool:
                nonlocal free_resources
                if not queue or not free_resources:
                    return False

                n = len(queue)
                for _ in range(n):
                    if not free_resources:
                        break

                    task = queue.popleft()

                    candidates = [r for r in free_resources if self.permissions.can_execute(r, task.activity)]
                    if not candidates:
                        queue.append(task)
                        continue

                    chosen = self.selector.select(candidates)
                    if chosen is None:
                        queue.append(task)
                        continue

                    free_resources.remove(chosen)
                    self._start_or_resume_task(t, task, chosen, resumed=resumed)
                    return True

                return False

            did_something = False
            if try_queue(self.suspended, resumed=True):
                did_something = True
            elif try_queue(self.waiting, resumed=False):
                did_something = True

            if not did_something:
                return

    # ---------- Export helpers ----------
    def _export_xes(self, df: pd.DataFrame, xes_path: str):
        if pm4py is None:
            print("⚠️ pm4py not installed -> skipping XES export.")
            return

        df2 = df.copy()
        df2["time:timestamp"] = pd.to_datetime(df2["time:timestamp"], utc=True, errors="coerce")
        df2 = df2.dropna(subset=["time:timestamp"]).copy()

        df2 = pm4py.format_dataframe(
            df2,
            case_id="case:concept:name",
            activity_key="concept:name",
            timestamp_key="time:timestamp",
        )

        log = pm4py.convert_to_event_log(df2)
        pm4py.write_xes(log, xes_path)
        print("✅ wrote XES:", xes_path)

    # ---------- Run ----------
    def run(self, max_cases: Optional[int] = None) -> str:
        t0 = self.start_time

        # schedule first arrival + first boundary check
        self._push_event(t0, EventType.CASE_ARRIVAL, {})
        self._push_event(self._next_bucket_boundary(t0), EventType.RESOURCE_CHECK, {})

        while True:
            ev = self._pop_event()
            if ev is None:
                break

            t = ev.ts
            if t > self.end_time:
                break

            if ev.type == EventType.CASE_ARRIVAL:
                if max_cases is not None and self.case_counter >= max_cases:
                    continue

                case_id = self._new_case_id()
                attrs = self.attr_sampler.sample()

                self.case_attrs[case_id] = attrs
                self.case_last2[case_id] = (None, None)

                if hasattr(self.bpmn, "start_activity"):
                    first_act = self.bpmn.start_activity(attrs)
                else:
                    first_act = "A_Create Application"

                # ✅ FIX: DO NOT log "case_arrival"
                # Only enqueue -> which logs "schedule"
                self._enqueue_waiting(t, Task(case_id=case_id, activity=first_act, ready_ts=t))

                # schedule next case arrival
                t_next = self.arrivals.next_arrival_time(t)
                if pd.Timestamp(t_next).tz_convert("UTC") <= self.end_time:
                    self._push_event(t_next, EventType.CASE_ARRIVAL, {})

                # dispatch
                self._push_event(t, EventType.RECHECK, {})

            elif ev.type == EventType.ACTIVITY_COMPLETE:
                self._complete_task(t, ev.payload.get("task_id"))

            elif ev.type == EventType.RESOURCE_CHECK:
                self._resource_check(t)

            elif ev.type == EventType.RECHECK:
                self._allocate(t)

        # Write CSV
        out_csv = Path(self.out_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        df_out = pd.DataFrame(self.log_rows)
        df_out.to_csv(out_csv, index=False)
        print("✅ wrote CSV:", str(out_csv))

        # Write XES (optional)
        if self.out_xes_path:
            out_xes = Path(self.out_xes_path)
            out_xes.parent.mkdir(parents=True, exist_ok=True)
            self._export_xes(df_out, str(out_xes))

        return str(out_csv)


# Optional tiny BPMN for quick testing
class SimpleBPMN:
    def __init__(self, edges: Dict[str, List[str]], start: str, finals: Optional[Set[str]] = None):
        self.edges = edges
        self.start = start
        self.finals = finals or set()

    def start_activity(self, case_attrs: Dict[str, Any]) -> str:
        return self.start

    def allowed_next(self, current_activity: str, case_attrs: Dict[str, Any]) -> List[str]:
        return list(self.edges.get(current_activity, []))

    def is_final(self, activity: str) -> bool:
        return activity in self.finals
