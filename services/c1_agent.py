"""
Component 1 -- Adaptive eBPF endpoint agent with runtime tracing depth.
IT23184312 | Amantha M A

What is real here and what is stubbed
------------------------------------
REAL   : the rarity scorer, the EWMA confidence model, the hysteresis + dwell
         mode controller, the E(t) external-alert path, the LSM enforcement
         handler, and the duty-cycle accounting. This is the research logic and
         it is the part being evaluated.
STUBBED: the kernel itself. Events arrive from a KernelSource. Today that is
         SyntheticKernelSource; on Ubuntu it becomes EbpfKernelSource reading a
         BPF ring buffer. The controller above it does not change.

The novelty being demonstrated: tracing depth is a runtime variable driven by a
live confidence score, not a deployment-time setting.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol

from laax.backend import SimulatedBackend, TracingBackend
from laax.bus import Bus
from laax.contracts import (
    EnforceHost,
    ExternalAlert,
    HostEvent,
    ModeChange,
    Topic,
)

# Probes attached only in deep mode. Loaded and verified at start-up, attached
# on demand -- deferred attachment is what keeps the switch inside 500 ms.
DEEP_PROBES = [
    "kprobe/security_bprm_check",
    "kretprobe/do_execveat_common",
    "kprobe/tcp_sendmsg",
    "uretprobe/libc:getaddrinfo",
]


# --------------------------------------------------------------------------
# Rarity scoring:  r = -log2( (count + a) / (total + a*N) )
# --------------------------------------------------------------------------
class RarityScorer:
    """
    Information-theoretic rarity over a benign baseline profile.

    Two contexts are tracked, both cheap enough to live in a BPF LRU hash in the
    real agent:
        lineage : (parent_comm -> child_comm)
        egress  : (comm -> dport)

    The baseline is built from BENIGN traffic only. Including attack data would
    normalise the attack pattern and flatten its rarity -- proposal slide 8.
    """

    def __init__(self, alpha: float = 0.5, vocab: int = 128) -> None:
        self.alpha, self.vocab = alpha, vocab
        self.counts: Dict[str, Dict[tuple, int]] = {"lineage": defaultdict(int),
                                                    "egress": defaultdict(int)}
        self.totals: Dict[str, int] = defaultdict(int)
        self.learning = True
        self._seen: list[float] = []
        self.baseline_rarity = 0.0     # mean rarity over the benign window

    def observe(self, ctx: str, key: tuple) -> None:
        self.counts[ctx][key] += 1
        self.totals[ctx] += 1

    def rarity(self, ctx: str, key: tuple) -> float:
        c = self.counts[ctx].get(key, 0)
        p = (c + self.alpha) / (self.totals[ctx] + self.alpha * self.vocab)
        return -math.log2(max(p, 1e-12))

    def score_event(self, ev: HostEvent) -> float:
        if ev.kind == "exec":
            key = (ev.comm_parent or "?", ev.comm)
            r = self.rarity("lineage", key)
            if self.learning:
                self.observe("lineage", key)
                self._seen.append(r)
            return r
        if ev.kind in ("connect", "accept"):
            key = (ev.comm, ev.dport)
            r = self.rarity("egress", key)
            if self.learning:
                self.observe("egress", key)
                self._seen.append(r)
            return r
        return 0.0

    def freeze(self) -> None:
        """End the baseline window. From here on, rare things stay rare."""
        self.learning = False
        if self._seen:
            self.baseline_rarity = sum(self._seen) / len(self._seen)
        self._seen = []


# HostEvent has no parent comm field in the wire contract (the kernel gives us
# ppid, and the agent resolves the name locally), so the agent carries it here.
HostEvent.comm_parent = None  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Per-process confidence and mode state
# --------------------------------------------------------------------------
@dataclass
class ProcState:
    pid: int
    comm: str
    confidence: float = 0.0
    mode: str = "light"
    entered_mode_at: float = field(default_factory=time.time)
    deep_seconds: float = 0.0
    events: int = 0


class ModeController:
    """
    Hysteresis + dwell.

    A single threshold makes the agent flap: the score oscillates around theta,
    probes attach and detach many times a second, and both the overhead figure
    and the event stream become garbage. Two thresholds plus a minimum dwell
    time in deep mode remove that failure entirely.
    """

    def __init__(
        self,
        theta_up: float = 7.0,
        theta_down: float = 4.0,
        min_dwell_s: float = 3.0,
        decay: float = 0.6,
        inherit: float = 0.7,
        attach_cost_ms: float = 3.2,
        map_update_ms: float = 0.04,
    ) -> None:
        assert theta_down < theta_up, "hysteresis requires theta_down < theta_up"
        self.theta_up, self.theta_down = theta_up, theta_down
        self.min_dwell_s, self.decay = min_dwell_s, decay
        self.inherit = inherit
        self.attach_cost_ms = attach_cost_ms
        self.map_update_ms = map_update_ms
        self.switches = 0
        # Attach and detach are kept apart because they are not the same cost:
        # detaching a kprobe waits out an RCU grace period and measures ~13x an
        # attach. Averaging them together reports a number that describes
        # neither operation.
        self.attach_latencies: list[float] = []   # light -> deep, global
        self.detach_latencies: list[float] = []   # deep -> light, global
        self.map_latencies: list[float] = []      # per-pid gating updates

    def update_confidence(self, st: ProcState, rarity: float,
                          baseline: float = 0.0, boost: float = 0.0) -> None:
        """
        CUSUM over rarity excess, not an EWMA.

        An EWMA averages, and averaging is wrong for this. Its fixed point under
        damped inheritance is inherit*decay-discounted rarity -- with the tuned
        values, 0.69 * rarity -- so a chain of maximally rare execs converges
        BELOW theta_up and can never fire, however long the chain. A real kernel
        exposed this immediately: 270 of 277 processes lived for one event.

        Evidence along a provenance chain accumulates, so the statistic should
        too. Each event contributes its surprisal in excess of what the benign
        baseline already explains; benign activity contributes about zero and a
        genuinely rare step contributes the difference. Clamped at zero, this is
        a standard one-sided CUSUM change detector.
        """
        st.confidence = max(0.0, st.confidence + (rarity - baseline) + boost)

    def decide(self, st: ProcState) -> Optional[tuple[str, str]]:
        """Return (to_mode, trigger) if a transition should happen."""
        now = time.time()
        held = now - st.entered_mode_at
        if st.mode == "light" and st.confidence >= self.theta_up:
            return ("deep", "rarity")
        if st.mode == "deep":
            if held < self.min_dwell_s:
                return None                      # dwell floor: refuse to flap
            if st.confidence <= self.theta_down:
                return ("light", "dwell_expiry")
        return None

    def record(self, st: ProcState, to_mode: str, latency_ms: float,
               global_transition: bool) -> None:
        """
        Commit a transition the backend has already performed, and book the
        latency into the right bucket.

        Why the two buckets matter for the overhead claim: a kprobe is attached
        ONCE, globally, and gated by a BPF hash of pids currently in deep mode.
        The expensive operation -- bpf_link creation against an already-verified,
        already-loaded program -- is paid only when the agent goes from zero deep
        processes to one, and again when it drops back to zero. Promoting a
        further process is a single bpf_map_update_elem.

        Collapsing these into one flat "attach cost per process" would overstate
        the cost of the adaptive design by roughly two orders of magnitude --
        against your own argument.
        """
        now = time.time()
        if st.mode == "deep":
            st.deep_seconds += now - st.entered_mode_at
        st.mode, st.entered_mode_at = to_mode, now
        self.switches += 1
        if global_transition:
            bucket = self.attach_latencies if to_mode == "deep" else self.detach_latencies
        else:
            bucket = self.map_latencies
        bucket.append(latency_ms)


# --------------------------------------------------------------------------
# Kernel sources
# --------------------------------------------------------------------------
class KernelSource(Protocol):
    def __aiter__(self): ...
    async def __anext__(self) -> HostEvent: ...


# --------------------------------------------------------------------------
class C1Agent:
    def __init__(self, bus: Bus, source, host_id: str = "host-0",
                 controller: Optional[ModeController] = None,
                 backend: Optional[TracingBackend] = None) -> None:
        self.bus, self.source, self.host_id = bus, source, host_id
        # SimulatedBackend models the costs; EbpfBackend measures them. The
        # agent cannot tell the difference, which is the point.
        self.backend: TracingBackend = backend or SimulatedBackend()
        self.scorer = RarityScorer()
        self.ctl = controller or ModeController()
        self.procs: Dict[int, ProcState] = {}
        self.comm_by_pid: Dict[int, str] = {}
        self.denied: Dict[int, set] = defaultdict(set)   # LSM enforcement state
        self.deep_pids: set[int] = set()                 # the BPF gating map
        self.probes_attached = False                     # global kprobe links
        self.deep_window_start: Optional[float] = None
        self.system_deep_seconds = 0.0                   # wall time with probes live
        self.started = time.time()
        self.emitted = 0
        self.suppressed_deep = 0                          # payloads dropped in light mode
        self.captured_deep = 0                            # payloads recovered in deep mode
        bus.subscribe(Topic.EXTERNAL_ALERT, self.on_external_alert)
        bus.subscribe(Topic.ENFORCE_HOST, self.on_enforce)

    # -- inbound from C3 ---------------------------------------------------
    async def on_external_alert(self, _topic: str, msg: ExternalAlert) -> None:
        if msg.host_id != self.host_id:
            return
        st = self.procs.get(msg.pid)
        if st is None:
            return
        before = st.confidence
        st.confidence += msg.boost
        print(f"  [C1] E(t) from C3: pid {msg.pid} confidence "
              f"{before:.2f} -> {st.confidence:.2f}  ({msg.reason})")
        await self._maybe_switch(st, forced_trigger="external_alert")

    # -- inbound from C4 ---------------------------------------------------
    async def on_enforce(self, _topic: str, msg: EnforceHost) -> None:
        if msg.host_id != self.host_id:
            return
        self.denied[msg.pid].add(msg.syscall)
        # Writes the pid into denied_pids, the map read by the
        # lsm/bprm_check_security program that returns -EPERM.
        latency = await self.backend.deny_add(msg.pid, msg.syscall)
        print(f"  [C1] LSM ARMED: deny {msg.syscall} for pid {msg.pid} "
              f"({latency:.3f}ms, lease {msg.lease_seconds:.0f}s, "
              f"policy {msg.policy_id})")
        # The lease is C4's contract; C1 honours it locally so enforcement can
        # never outlive the policy that authorised it.
        asyncio.create_task(self._expire_denial(msg.pid, msg.syscall,
                                                msg.lease_seconds))

    async def _expire_denial(self, pid: int, syscall: str, after: float) -> None:
        await asyncio.sleep(after)
        if syscall in self.denied.get(pid, ()):  
            self.denied[pid].discard(syscall)
            await self.backend.deny_del(pid)
            print(f"  [C1] LSM RELEASED: {syscall} allowed again for pid {pid}")

    # -- mode switching ----------------------------------------------------
    async def _switch(self, st: ProcState, to_mode: str, trigger: str) -> None:
        from_mode = st.mode
        # Does this transition change whether the kprobes are attached at all?
        will_be_deep = (self.deep_pids | {st.pid}) if to_mode == "deep" else (
            self.deep_pids - {st.pid})
        global_transition = bool(self.deep_pids) != bool(will_be_deep)

        # Gate the pid first, then attach if this is the first deep process --
        # in that order, so the probe can never fire for a pid the map does not
        # yet contain. On the way out, detach after the last pid is removed.
        if to_mode == "deep":
            latency = await self.backend.gate_add(st.pid)
            if global_transition:
                latency += await self.backend.attach_deep()
        else:
            latency = await self.backend.gate_del(st.pid)
            if global_transition:
                latency += await self.backend.detach_deep()

        self.ctl.record(st, to_mode, latency, global_transition)
        self.deep_pids = will_be_deep

        if global_transition:
            self.probes_attached = to_mode == "deep"
            now = time.time()
            if self.probes_attached:
                self.deep_window_start = now
            elif self.deep_window_start is not None:
                self.system_deep_seconds += now - self.deep_window_start
                self.deep_window_start = None

        kind = "kprobe attach" if global_transition and to_mode == "deep" else (
            "kprobe detach" if global_transition else "map update")
        print(f"  [C1] {'>>' if to_mode == 'deep' else '<<'} pid {st.pid} ({st.comm}) "
              f"{from_mode} -> {to_mode.upper() if to_mode == 'deep' else to_mode}  "
              f"conf={st.confidence:.2f}  {kind}={latency:.3f}ms  "
              f"[{trigger}]")
        await self.bus.publish(Topic.HOST_MODE, ModeChange(
            host_id=self.host_id, pid=st.pid, comm=st.comm,
            from_mode=from_mode, to_mode=to_mode,
            confidence=round(st.confidence, 3),
            threshold=self.ctl.theta_up if to_mode == "deep" else self.ctl.theta_down,
            trigger=trigger,
            attach_latency_ms=round(latency, 3),
            probes=DEEP_PROBES if to_mode == "deep" else [],
        ))

    async def _maybe_switch(self, st: ProcState, forced_trigger: str = "rarity") -> None:
        # The learning window is light-mode by construction. Rarity is defined
        # relative to a benign baseline, so before that baseline exists the
        # confidence score has no meaning and must not drive anything: early in
        # the window every event looks rare simply because little has been
        # counted yet. Acting on it produced 23 spurious deep-mode entries
        # during a benign phase that must, by the overhead claim, produce zero.
        #
        # An external alert from C3 is exempt: that evidence does not come from
        # our own uncalibrated baseline.
        if self.scorer.learning and forced_trigger != "external_alert":
            return
        decision = self.ctl.decide(st)
        if decision is None:
            return
        to_mode, trigger = decision
        await self._switch(st, to_mode,
                           forced_trigger if forced_trigger != "rarity" else trigger)

    # -- main loop ---------------------------------------------------------
    async def run(self) -> None:
        async for ev in self.source:
            self.comm_by_pid[ev.pid] = ev.comm
            # The kernel gives us the parent comm directly on exec (see
            # laax.bpf.c). Only fall back to a local pid->comm lookup when it
            # did not. Overwriting it unconditionally collapsed every lineage
            # key to ('?', child) and destroyed the rarity signal.
            if not getattr(ev, "comm_parent", None):
                ev.comm_parent = self.comm_by_pid.get(ev.ppid)  # type: ignore[attr-defined]

            st = self.procs.get(ev.pid)
            if st is None:
                # Seed a new process from its parent rather than from zero.
                #
                # Without this the agent cannot react to short-lived processes
                # at all: a process that execs once and exits gets a single
                # EWMA update, which from a standing start cannot reach
                # theta_up however rare the event is. On a real kernel that is
                # not an edge case -- 270 of 277 observed processes emitted
                # exactly one event -- and `whoami; uname; id; nc` is precisely
                # that shape. Suspicion is a property of the process TREE, so
                # it has to flow down the tree the way provenance does.
                parent = self.procs.get(ev.ppid)
                seed = parent.confidence * self.ctl.inherit if parent else 0.0
                st = ProcState(pid=ev.pid, comm=ev.comm, confidence=seed)
                self.procs[ev.pid] = st
                # Deep mode is inherited down the process tree. A pivot chain is
                # a subtree, not a single pid, and a child spawned by a process
                # under investigation must not be born blind.
                if ev.ppid in self.deep_pids:
                    await self._switch(st, "deep", "inherited")
            st.events += 1

            rarity = self.scorer.score_event(ev)
            self.ctl.update_confidence(st, rarity, self.scorer.baseline_rarity)

            ev.rarity = round(rarity, 3)
            ev.confidence = round(st.confidence, 3)
            # The kernel source says which hook fired. A scripted source does
            # not, so infer it: only a deep-mode process can have its payload
            # captured, and only the deep kprobes capture one.
            if ev.hook is None:
                ev.hook = "kprobe" if st.mode == "deep" and ev.argv else "tracepoint"
            ev.depth = st.mode
            ev.host_id = self.host_id

            # Light mode carries no forensic payload. This is the cost saving,
            # and equally the visibility loss the adaptive design exists to fix.
            if st.mode == "light":
                if ev.argv or ev.syscall:
                    self.suppressed_deep += 1
                ev.argv, ev.syscall, ev.ret = None, None, None
            elif ev.argv:
                self.captured_deep += 1
                print(f"  [C1] deep payload  pid {ev.pid} ({ev.comm}) "
                      f"argv={' '.join(ev.argv)}")

            await self.bus.publish(Topic.HOST_EVENT, ev)
            self.emitted += 1
            await self._maybe_switch(st)

    # -- time-based decay --------------------------------------------------
    async def decay_loop(self, tick: float = 0.5) -> None:
        """
        Confidence must decay on a clock, not only on events.

        Without this the agent has a one-way ratchet: a process that triggers
        deep mode and then goes quiet produces no further events, so its EWMA is
        never updated, so it never falls back to light. Duty cycle then rises
        monotonically to 100% and the entire overhead argument collapses. This
        loop is what makes the mechanism genuinely bidirectional.
        """
        while True:
            await asyncio.sleep(tick)
            for st in list(self.procs.values()):
                if st.mode != "deep":
                    continue
                st.confidence *= self.ctl.decay
                decision = self.ctl.decide(st)
                if decision is not None:
                    await self._switch(st, decision[0], "decay")

    # -- evaluation metrics ------------------------------------------------
    def stats(self) -> dict:
        now = time.time()
        wall = max(now - self.started, 1e-9)
        deep = self.system_deep_seconds + (
            (now - self.deep_window_start) if self.deep_window_start else 0.0)
        lat = self.ctl.attach_latencies
        dlat = self.ctl.detach_latencies
        mlat = self.ctl.map_latencies
        return {
            "events_emitted": self.emitted,
            "processes_tracked": len(self.procs),
            "mode_switches": self.ctl.switches,
            "kprobe_attach_events": len(lat),
            "kprobe_detach_events": len(dlat),
            # The decisive metric (proposal slide 8): fraction of wall time the
            # deep probes were attached at all. At 100% this reduces to
            # always-deep and the contribution evaporates.
            "duty_cycle_pct": round(100.0 * deep / wall, 2),
            "attach_latency_ms_mean": round(sum(lat) / len(lat), 3) if lat else 0.0,
            "attach_latency_ms_max": round(max(lat), 3) if lat else 0.0,
            "detach_latency_ms_mean": round(sum(dlat) / len(dlat), 3) if dlat else 0.0,
            "detach_latency_ms_max": round(max(dlat), 3) if dlat else 0.0,
            "worst_switch_ms": round(max(lat + dlat), 3) if (lat or dlat) else 0.0,
            "map_update_ms_mean": round(sum(mlat) / len(mlat), 4) if mlat else 0.0,
            "deep_payloads_suppressed": self.suppressed_deep,
            "deep_payloads_captured": self.captured_deep,
            "lsm_denials_armed": sum(len(v) for v in self.denied.values()),
            "latency_source": "measured" if getattr(self.backend, "measured", False) else "modelled",
        }
