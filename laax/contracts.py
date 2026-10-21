"""
LAAX canonical message contracts.

This module is the integration seam of the whole framework. Every component
speaks these types and nothing else, so any component can be replaced by a
mock, a baseline, or the real artefact without touching its neighbours.

THE CENTRAL DESIGN DECISION -- read this before changing HostEvent
-------------------------------------------------------------------
Component 1 produces eBPF syscall events. Component 3 trains on TON_IoT Linux
host telemetry, which is *periodically sampled system metrics* (process/memory/
disk counters), not a syscall event stream. Those two data models do not match,
so C1 live output cannot be fed to a C3 model trained on raw TON_IoT.

HostEvent resolves this. It is a canonical host-domain record that BOTH sides
can produce:

    eBPF agent  --(native)------> HostEvent  <--(adapter)-- TON_IoT Linux CSV

C3 trains on adapted TON_IoT and runs live on C1 output, because both arrive as
HostEvent. The adapter lives in services/ton_iot_adapter.py. Fields that only
one producer can fill are Optional and every consumer must tolerate None.
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "0.1.0"


# --------------------------------------------------------------------------
# Topics
# --------------------------------------------------------------------------
class Topic:
    HOST_EVENT = "host.event"          # C1 -> C3
    HOST_MODE = "host.mode"            # C1 -> console  (observability)
    FLOW_VERDICT = "iot.flow.verdict"  # C2 -> C3
    CORRELATION = "corr.alert"         # C3 -> C4
    EXTERNAL_ALERT = "corr.external"   # C3 -> C1       (the E(t) feedback arrow)
    CONTAINMENT = "resp.action"        # C4 -> console / enforcement arms
    ENFORCE_HOST = "resp.enforce.host" # C4 -> C1       (LSM syscall denial)
    ROLLBACK = "resp.rollback"         # C4 -> console


def _now() -> float:
    return time.time()


# Message ids: a random per-process prefix plus a counter. Unique across hosts
# and restarts like a UUID for every practical purpose, without a getrandom()
# syscall per telemetry event (uuid4 was a measurable share of per-event cost).
_ID_PREFIX = os.urandom(4).hex()
_ID_COUNTER = itertools.count()


def _mid() -> str:
    return f"{_ID_PREFIX}{next(_ID_COUNTER):x}"


_FIELD_NAMES: Dict[type, tuple] = {}


def _names(cls: type) -> tuple:
    names = _FIELD_NAMES.get(cls)
    if names is None:
        names = _FIELD_NAMES[cls] = tuple(f.name for f in fields(cls))
    return names


@dataclass
class Message:
    """Common envelope. Subclasses add a payload."""
    msg_id: str = field(default_factory=_mid)
    ts: float = field(default_factory=_now)
    schema: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        # One level of copying is all the contracts need: fields are scalars,
        # lists of scalars, or (CorrelationAlert.path) a list of flat dicts.
        # dataclasses.asdict deep-copies recursively and cost ~20 us per event.
        out = {}
        for name in _names(type(self)):
            v = getattr(self, name)
            if isinstance(v, list):
                v = [dict(x) if isinstance(x, dict) else x for x in v]
            elif isinstance(v, dict):
                v = dict(v)
            out[name] = v
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------
# Component 1 -> Component 3
# --------------------------------------------------------------------------
@dataclass
class HostEvent(Message):
    """One host-domain observation, from eBPF or from adapted TON_IoT."""

    host_id: str = "host-0"
    # -- identity, always present -----------------------------------------
    #
    # pid and ppid are ROOT PID NAMESPACE pids, because that is what
    # bpf_get_current_pid_tgid() returns. They are NOT what a process inside a
    # container or a WSL distro sees for itself: measured on this machine, one
    # process was pid 389 to its own namespace and 4984 to the kernel.
    #
    # Consequence for every consumer: a pid in a LAAX message is only meaningful
    # if it originated from C1. Never populate one from os.getpid(), from /proc
    # inside a container, or from an operator typing a pid they read in `ps`.
    # C4's EnforceHost.pid in particular must be a pid C1 reported first, or the
    # LSM map lookup silently never matches and the containment quietly no-ops.
    pid: int = 0
    ppid: int = 0
    comm: str = ""
    # Path of the binary the kernel executed (exec events). comm is set by the
    # process itself and can be anything; exe cannot be chosen that way.
    exe: Optional[str] = None
    # -- event classification ---------------------------------------------
    kind: str = "exec"           # exec | connect | accept | file | sample
    # Two different facts, kept apart. `depth` is the PROCESS's tracing mode when
    # C1 emitted the event (set by the agent). `hook` is the kernel attachment
    # that produced it. A deep-mode process still trips the light tracepoints,
    # so a deep process can emit a tracepoint event; collapsing the two would
    # report kprobe coverage that never existed.
    depth: str = "light"         # light | deep
    hook: Optional[str] = None   # tracepoint | kprobe | uretprobe | xdp
    # -- network context, present on connect/accept ------------------------
    saddr: Optional[str] = None
    daddr: Optional[str] = None
    sport: Optional[int] = None  # None when the hook fires before a port is bound
    dport: Optional[int] = None
    proto: Optional[str] = None
    # -- deep-mode only ----------------------------------------------------
    argv: Optional[List[str]] = None
    syscall: Optional[str] = None
    ret: Optional[int] = None
    # -- sampled metrics: the fields TON_IoT can fill, eBPF leaves None ----
    cpu_pct: Optional[float] = None
    rss_kb: Optional[int] = None
    # -- scoring -----------------------------------------------------------
    rarity: float = 0.0          # -log2 p(event | benign baseline)
    confidence: float = 0.0      # per-process confidence at emit time
    source: str = "ebpf"         # ebpf | ton_iot | synthetic
    label: Optional[str] = None  # ground truth when replaying a dataset


@dataclass
class ModeChange(Message):
    """C1 tracing-depth transition. The core novelty, made observable."""

    host_id: str = "host-0"
    pid: int = 0
    comm: str = ""
    from_mode: str = "light"
    to_mode: str = "deep"
    confidence: float = 0.0
    threshold: float = 0.0
    trigger: str = "rarity"      # rarity | external_alert | dwell_expiry
    attach_latency_ms: float = 0.0
    probes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Component 2 -> Component 3
# --------------------------------------------------------------------------
@dataclass
class FlowVerdict(Message):
    """Per-flow anomaly score for one unmanaged IoT device."""

    device_id: str = ""
    saddr: str = ""
    daddr: str = ""
    dport: int = 0
    proto: str = "tcp"
    score: float = 0.0           # P(malicious)
    verdict: str = "benign"      # benign | suspicious | malicious
    feature_set: str = "B"       # A = identifier-inclusive, B = behavioural-only
    model: str = "gbdt"
    label: Optional[str] = None


# --------------------------------------------------------------------------
# Component 3 -> Components 1 and 4
# --------------------------------------------------------------------------
@dataclass
class CorrelationAlert(Message):
    """A cross-domain lateral-movement path C3 believes is real."""

    alert_id: str = field(default_factory=_mid)
    severity: str = "medium"     # low | medium | high | critical
    confidence: float = 0.0
    path: List[Dict[str, Any]] = field(default_factory=list)  # ordered graph nodes
    iot_device: Optional[str] = None
    host_id: Optional[str] = None
    pid: Optional[int] = None
    technique: str = "lateral_movement"
    explanation: str = ""


@dataclass
class ExternalAlert(Message):
    """E(t): C3 raising C1 confidence for a specific process. Feedback arrow."""

    host_id: str = "host-0"
    pid: int = 0
    boost: float = 0.0           # added to the process confidence score
    reason: str = ""
    alert_id: str = ""


# --------------------------------------------------------------------------
# Component 4
# --------------------------------------------------------------------------
@dataclass
class ContainmentAction(Message):
    """A graduated, reversible response. Every action carries a lease."""

    policy_id: str = field(default_factory=_mid)
    alert_id: str = ""
    level: int = 0               # 0 observe | 1 rate-limit | 2 segment | 3 isolate
    layer: str = "sdn"           # sdn | ebpf | both
    target: str = ""             # device id or host:pid
    action: str = ""             # human-readable
    lease_seconds: float = 60.0
    rationale: str = ""          # explainability record
    reversible: bool = True


@dataclass
class EnforceHost(Message):
    """C4 -> C1: deny a specific syscall for a specific process via eBPF LSM."""

    policy_id: str = ""
    host_id: str = "host-0"
    pid: int = 0
    syscall: str = "execve"
    lease_seconds: float = 60.0


@dataclass
class Rollback(Message):
    """Containment withdrawn: lease expired, or the evidence was disproved."""

    policy_id: str = ""
    reason: str = "lease_expired"   # lease_expired | disproved | analyst_override
    target: str = ""
    success: bool = True


# --------------------------------------------------------------------------
DECODERS = {
    Topic.HOST_EVENT: HostEvent,
    Topic.HOST_MODE: ModeChange,
    Topic.FLOW_VERDICT: FlowVerdict,
    Topic.CORRELATION: CorrelationAlert,
    Topic.EXTERNAL_ALERT: ExternalAlert,
    Topic.CONTAINMENT: ContainmentAction,
    Topic.ENFORCE_HOST: EnforceHost,
    Topic.ROLLBACK: Rollback,
}


def decode(topic: str, payload: Dict[str, Any]) -> Message:
    return DECODERS[topic].from_dict(payload)
