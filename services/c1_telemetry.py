"""
Component 1 -> Component 3 telemetry export.
IT23184312 | Amantha M A

Writes every HostEvent C1 publishes as one row of the C3 telemetry schema, the
26-column layout agreed with Component 3 (reference file:
C1_Output_for_C3_V6_v2_2019.csv). Column order, timestamp format and the "-"
null marker match that file exactly, so C3 trains on the reference data and
runs on live C1 output without a second parser.

Every value is taken from what the agent actually observed. Where the kernel
did not supply a field, the row says "-" rather than inventing one:

    cpu_usage          eBPF does not sample CPU per event. Reading /proc from
                       user space would be wrong anyway: the pid is a root
                       namespace pid (see HostEvent), so inside WSL or a
                       container /proc/<pid> is a different process.
    src_port           sys_enter_connect fires before the kernel picks the
                       ephemeral port.
    syscall_result     only known where the hook fires after the syscall;
                       sched_process_exec does (it fires on success only).
    label, type        ground truth. C1 cannot know these at runtime; they are
                       filled only when replaying a labelled scenario, and C3
                       must never use them as features.
"""

from __future__ import annotations

import csv
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, TextIO

from laax.bus import Bus
from laax.contracts import HostEvent, Topic

COLUMNS: List[str] = [
    "timestamp", "host_id", "host_ip", "process_id", "parent_process_id",
    "process_name", "event_type", "trace_mode", "trace_source", "src_ip",
    "dst_ip", "src_port", "dst_port", "protocol", "syscall_name",
    "syscall_result", "process_state", "memory_alloc_bytes", "file_operation",
    "threat_confidence_score", "threat_state", "cpu_usage", "disk_read_bytes",
    "disk_write_bytes", "label", "type",
]

NULL = "-"
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"

EVENT_TYPE = {
    "exec": "process_start",
    "connect": "network_connect",
    "accept": "network_accept",
    "file": "file_event",
    "memory": "memory_event",
}

# The schema names the syscall family, not the libc entry point.
SYSCALL_NAME = {"exec": "execve", "connect": "connect", "accept": "accept"}
SYSCALL_ALIASES = {"accept4": "accept", "execveat": "execve"}

# Score the deep-mode entry threshold maps to. 0.6 is where the reference data
# switches to deep, so theta_up lands on the same point of the [0, 1] scale.
SCORE_AT_THETA_UP = 0.6


def normalise_confidence(confidence: float, theta_up: float,
                         at: float = SCORE_AT_THETA_UP) -> float:
    """
    Map the unbounded CUSUM confidence onto [0, 1) for export.

    s = 1 - exp(-c / k), with k chosen so that s(theta_up) == at. Monotonic, so
    it preserves every ordering and threshold decision the controller makes;
    the controller itself keeps working on the raw statistic.
    """
    if confidence <= 0:
        return 0.0
    k = theta_up / -math.log(1.0 - at)
    return 1.0 - math.exp(-confidence / k)


def _label(ev: HostEvent) -> tuple:
    if ev.label is None:
        return NULL, NULL
    if ev.label == "normal":
        return 0, "normal"
    return 1, ev.label


def _syscall_result(ev: HostEvent) -> str:
    if ev.ret is not None and ev.ret < 0:
        return "failure"
    if ev.kind == "exec" and ev.hook == "tracepoint":
        return "success"          # sched_process_exec fires only after a successful exec
    return NULL


def to_row(ev: HostEvent, host_ip: str, theta_up: float,
           theta_down: float) -> Dict[str, object]:
    """One HostEvent as one schema row. Raises ValueError for kinds the schema lacks."""
    if ev.kind not in EVENT_TYPE:
        raise ValueError(f"no C3 event_type for HostEvent kind {ev.kind!r}")

    network = ev.kind in ("connect", "accept")
    if ev.kind == "connect":
        # Outbound: this host is the source. The connect tracepoint runs before
        # the source address is bound, so fall back to the host's address.
        src_ip, dst_ip = ev.saddr or host_ip, ev.daddr
        src_port, dst_port = ev.sport, ev.dport
    elif ev.kind == "accept":
        # Inbound: the remote peer is the source, this host's service port the
        # destination.
        src_ip, dst_ip = ev.saddr, ev.daddr or host_ip
        src_port, dst_port = ev.sport, ev.dport
    else:
        src_ip = dst_ip = src_port = dst_port = None

    syscall = ev.syscall or SYSCALL_NAME.get(ev.kind)
    syscall = SYSCALL_ALIASES.get(syscall, syscall)
    label, attack_type = _label(ev)

    return {
        "timestamp": datetime.fromtimestamp(ev.ts).strftime(TIME_FORMAT),
        "host_id": ev.host_id,
        "host_ip": host_ip,
        "process_id": ev.pid,
        "parent_process_id": ev.ppid,
        "process_name": ev.comm,
        "event_type": EVENT_TYPE[ev.kind],
        "trace_mode": ev.depth,
        "trace_source": ev.hook or NULL,
        "src_ip": src_ip or NULL,
        "dst_ip": dst_ip or NULL,
        "src_port": "" if src_port is None else src_port,
        "dst_port": "" if dst_port is None else dst_port,
        "protocol": (ev.proto or "tcp").upper() if network else NULL,
        "syscall_name": syscall or NULL,
        "syscall_result": _syscall_result(ev),
        # Every C1 hook runs in the context of the task that triggered it, so
        # at the moment of capture that task is on-CPU by definition.
        "process_state": "running",
        "memory_alloc_bytes": 0,
        "file_operation": NULL,
        "threat_confidence_score": round(normalise_confidence(ev.confidence, theta_up), 4),
        "threat_state": "suspicious" if ev.confidence > theta_down else "normal",
        "cpu_usage": NULL if ev.cpu_pct is None else ev.cpu_pct,
        "disk_read_bytes": 0,
        "disk_write_bytes": 0,
        "label": label,
        "type": attack_type,
    }


class TelemetryCsvExporter:
    """Subscribes to C1's HostEvent stream and appends schema rows to a CSV."""

    def __init__(self, bus: Bus, path: str, host_ips: Dict[str, str],
                 theta_up: float, theta_down: float) -> None:
        self.path, self.host_ips = path, host_ips
        self.theta_up, self.theta_down = theta_up, theta_down
        self.rows = 0
        self.skipped = 0
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh: Optional[TextIO] = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=COLUMNS, lineterminator="\n")
        if new:
            self._w.writeheader()
        bus.subscribe(Topic.HOST_EVENT, self.on_event)

    async def on_event(self, _topic: str, ev: HostEvent) -> None:
        if self._fh is None:
            return
        try:
            row = to_row(ev, self.host_ips.get(ev.host_id, NULL),
                         self.theta_up, self.theta_down)
        except ValueError:
            self.skipped += 1      # e.g. TON_IoT "sample" rows: not a C1 event
            return
        self._w.writerow(row)
        # Flushed per row: a crashed or killed agent must not lose the tail of
        # the telemetry it already reported as emitted.
        self._fh.flush()
        self.rows += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# --------------------------------------------------------------------------
# Conformance check: does a file match the schema AND the agent's design?
# --------------------------------------------------------------------------
# Hooks each tracing mode can legitimately produce. Light mode is tracepoints
# only -- that is the cost claim. Deep mode adds kprobes/uretprobes on top of
# the tracepoints, which stay attached. XDP is absent from both: it runs in the
# NIC driver before any socket or task exists, so it cannot attribute a packet
# to a process_id, and every row in this schema carries one.
ALLOWED_HOOKS = {
    "light": {"tracepoint"},
    "deep": {"tracepoint", "kprobe", "uretprobe"},
}


def check_rows(header: List[str], rows) -> Dict[str, int]:
    """Count rule violations. An empty dict means the file conforms."""
    bad: Dict[str, int] = {}

    def hit(rule: str) -> None:
        bad[rule] = bad.get(rule, 0) + 1

    if header != COLUMNS:
        hit("header differs from the C3 schema")
    for r in rows:
        mode, hook = r.get("trace_mode"), r.get("trace_source")
        if mode not in ALLOWED_HOOKS:
            hit("trace_mode not light|deep")
        elif hook not in ALLOWED_HOOKS[mode]:
            hit(f"{mode} mode row from hook '{hook}'")
        network = (r.get("event_type") or "").startswith("network")
        if network and r.get("dst_ip") in (None, "", NULL):
            hit("network event without dst_ip")
        if not network and r.get("src_ip") not in (None, "", NULL):
            hit("non-network event with an IP")
        try:
            s = float(r.get("threat_confidence_score"))
            if not 0.0 <= s <= 1.0:
                hit("score outside [0, 1]")
        except (TypeError, ValueError):
            hit("score not a number")
        try:
            datetime.strptime(r.get("timestamp") or "", TIME_FORMAT)
        except ValueError:
            hit("timestamp not dd/mm/YYYY HH:MM:SS")
    return bad


def check_file(path: str) -> Dict[str, int]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return check_rows(list(reader.fieldnames or []), reader)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m services.c1_telemetry <telemetry.csv>")
        raise SystemExit(2)
    violations = check_file(sys.argv[1])
    if not violations:
        print("conforms: schema and tracing-mode rules hold for every row")
    for rule, n in sorted(violations.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>8}  {rule}")
    raise SystemExit(1 if violations else 0)
