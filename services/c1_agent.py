"""
Component 1 -- Adaptive eBPF endpoint agent with runtime tracing depth.
IT23184312 | Amantha M A

What is real here and what is stubbed
------------------------------------
REAL   : the explainable rarity scorer, the CUSUM confidence model, the hysteresis + dwell
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
import ipaddress
import math
import re
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
    Rollback,
    Topic,
)
from services.c1_audit import AuditLog

# Probes attached only in deep mode. Loaded and verified at start-up, attached
# on demand -- deferred attachment is what keeps the switch inside 500 ms.
# Must match the deep tier in ebpf/laax.bpf.c exactly: this list is published
# in every ModeChange, so an entry here that the kernel lacks is false telemetry.
DEEP_PROBES = [
    "kprobe/finalize_exec",          # exec with full argv
    "kprobe/tcp_sendmsg",            # outbound data: peer, ports, bytes
    "kretprobe/inet_csk_accept",     # inbound TCP connections
    "kprobe/security_file_open",     # file opens: name + flags
]

# The LSM hook is lsm/bprm_check_security, so execve is the only syscall C1 can
# deny. A request for anything else is refused, never armed as execve.
SUPPORTED_DENIALS = {"execve"}

# laax.h LAAX_F_*: what a forked child inherited in-kernel.
INHERIT_DEEP, INHERIT_DENY = 0x1, 0x2


# --------------------------------------------------------------------------
# Rarity scoring:  r = -log2( (count + a) / (total + a*N) )
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Signal:
    name: str
    context: str
    key: tuple
    rarity: float
    why: str


@dataclass(frozen=True)
class Score:
    """One evaluated event, reusable for guarded learning and score logging."""
    evidence: float
    native_score: float
    signals: tuple[Signal, ...]

    def explanation(self) -> Dict[str, object]:
        """Build the expensive analyst-facing form only when it is requested."""
        detail = {}
        for signal in self.signals:
            contribution = RarityScorer.SIGNAL_WEIGHTS[signal.name] * signal.rarity
            detail[signal.name] = {
                "rarity": round(signal.rarity, 4),
                "weight": RarityScorer.SIGNAL_WEIGHTS[signal.name],
                "contribution": round(contribution, 4),
                "why": signal.why,
                "key": list(signal.key),
            }
        return {"version": "v2", "signals": detail,
                "raw_evidence": round(self.evidence, 4),
                "native_score": round(self.native_score, 4)}


# --------------------------------------------------------------------------
# Process-name canonicalisation
# --------------------------------------------------------------------------
# Some comms carry a per-thread serial that is unique to every thread in a
# pool: Firefox names its resolver threads "DNS Res~er #648", "#646", ... and
# its socket workers rotate too. Keyed raw, every one is a never-before-seen
# token and scores as maximally rare, so the rarity signal ends up tracking
# "the browser opened a new worker" instead of anything adversarial. The ROC
# calibration of 22 Oct 2026 showed these dominated both the attack windows
# and the benign false-positive wall (~13k events), flattening v1 and v2 to an
# identical AUC ~0.70. Collapsing the volatile serial to the stable pool name
# lets the baseline learn it once, as one common key.
_THREAD_SERIAL = re.compile(r"\s*#\d+$")        # "DNS Res~er #648" -> "DNS Res~er"
_KWORKER = re.compile(r"^kworker/\S+$")          # kworker/u16:3-events -> kworker


def canon_comm(comm: Optional[str]) -> Optional[str]:
    """Collapse volatile per-thread serials so a thread pool is one rarity key.

    Deliberately conservative: it only removes a trailing ``#<n>`` serial and
    folds the kernel worker pool. It never touches version suffixes or paths,
    so genuinely distinct programs (``python3.11`` vs ``python3``) stay
    distinct and the adversarial signal is preserved.
    """
    if not comm:
        return comm
    canon = _THREAD_SERIAL.sub("", comm)
    if _KWORKER.match(canon):
        return "kworker"
    return canon


class RarityScorer:
    """
    Information-theoretic rarity over a benign baseline profile.

    Scoring v2 deliberately keeps each signal separate so an analyst can see
    *why* a process became suspicious:

        ancestry    : (grandparent -> parent -> child), falling back to the
                      one-step (parent -> child) key when the grandparent was
                      not observed;
        egress      : (process -> destination zone -> port);
        executable  : the path captured by sched_process_exec, not the mutable
                      process name.

    The baseline is built from BENIGN traffic only. Including attack data would
    normalise the attack pattern and flatten its rarity -- proposal slide 8.
    This uses **Lidstone smoothing** (alpha = 0.5), not Laplace smoothing.
    """

    SIGNAL_WEIGHTS = {"ancestry": 0.50, "egress": 0.30, "executable": 0.20}
    # A single ancestry signal should retain the same initial evidence scale as
    # v1, so already-calibrated CUSUM thresholds remain meaningful.
    RAW_SCALE = 2.0

    def __init__(self, alpha: float = 0.5, vocab: int = 128,
                 zones: Optional[Dict[str, list[str]]] = None) -> None:
        self.alpha, self.vocab = alpha, vocab
        # ``lineage`` is retained as an alias for compatibility with old score
        # logs and status output. New profiles use ancestry_2 when possible.
        self.counts: Dict[str, Dict[tuple, int]] = {
            "lineage": defaultdict(int), "ancestry_2": defaultdict(int),
            "egress": defaultdict(int), "executable": defaultdict(int),
        }
        self.totals: Dict[str, int] = defaultdict(int)
        self.learning = True
        self._baseline_samples: list[float] = []
        self._baseline_sum = 0.0
        self._baseline_count = 0
        self.baseline_rarity = 0.0     # mean rarity over the benign window
        self.calm_rarity = 0.0         # learned p95 at freeze, not a magic constant
        self.last_explanation: Dict[str, object] = {}
        self._zones: list[tuple[str, ipaddress._BaseNetwork]] = []
        for name, networks in (zones or {}).items():
            for network in networks:
                self._zones.append((name, ipaddress.ip_network(network, strict=False)))

    def destination_zone(self, address: Optional[str]) -> str:
        """Return the configured zone for an address without guessing one.

        Longest-prefix matching lets an explicit IoT subnet take precedence
        over a broader private-network rule. Addresses absent from a connect
        record stay ``unknown`` rather than being fabricated.
        """
        if not address:
            return "unknown"
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return "unknown"
        matches = [(net.prefixlen, name) for name, net in self._zones if ip in net]
        if matches:
            return max(matches)[1]
        if ip.is_loopback:
            return "loopback"
        if ip.is_link_local:
            return "link_local"
        if ip.is_private:
            return "private"
        return "public"

    def observe(self, ctx: str, key: tuple) -> None:
        self.counts[ctx][key] += 1
        self.totals[ctx] += 1

    def rarity(self, ctx: str, key: tuple) -> float:
        c = self.counts[ctx].get(key, 0)
        p = (c + self.alpha) / (self.totals[ctx] + self.alpha * self.vocab)
        return -math.log2(max(p, 1e-12))

    def evaluate(self, ev: HostEvent) -> Score:
        """Evaluate an event once; callers reuse this for learning and logging."""
        signals: list[Signal] = []
        if ev.kind == "exec":
            child = canon_comm(ev.comm)
            parent = canon_comm(getattr(ev, "comm_parent", None)) or "?"
            grandparent = canon_comm(getattr(ev, "comm_grandparent", None))
            if grandparent:
                ctx, key = "ancestry_2", (grandparent, parent, child)
                why = "two-step ancestry"
            else:
                ctx, key = "lineage", (parent, child)
                why = "one-step ancestry fallback"
            signals.append(Signal("ancestry", ctx, key, self.rarity(ctx, key), why))
            if ev.exe:
                key = (ev.exe,)
                signals.append(Signal("executable", "executable", key,
                                      self.rarity("executable", key),
                                      "executed binary path"))
        if ev.kind in ("connect", "accept"):
            zone = self.destination_zone(ev.daddr)
            key = (canon_comm(ev.comm), zone, ev.dport or 0)
            signals.append(Signal("egress", "egress", key,
                                  self.rarity("egress", key),
                                  f"destination zone {zone}"))
        weighted = sum(self.SIGNAL_WEIGHTS[s.name] * s.rarity for s in signals)
        evidence = weighted * self.RAW_SCALE
        native = 1.0 - math.exp(-evidence / 7.0) if evidence else 0.0
        return Score(evidence=evidence, native_score=native, signals=tuple(signals))

    def _observe_score(self, score: Score) -> bool:
        for signal in score.signals:
            self.observe(signal.context, signal.key)
        self._baseline_sum += score.evidence
        self._baseline_count += 1
        self.baseline_rarity = self._baseline_sum / self._baseline_count
        return bool(score.signals)

    def score_event(self, ev: HostEvent) -> float:
        """Compatibility API for tools; the live agent uses evaluate()."""
        score = self.evaluate(ev)
        self.last_explanation = score.explanation()
        if self.learning:
            self._baseline_samples.append(score.evidence)
            self._observe_score(score)
        return score.evidence

    def learn(self, score: Score, *, confidence_before: float, mode: str,
              confidence_limit: float) -> bool:
        """Adapt only below the ROC-selected de-escalation and baseline p95."""
        if self.learning:
            self._baseline_samples.append(score.evidence)
            return self._observe_score(score)
        # Online adaptation must never turn a previously unseen feature into
        # normal behaviour on its first appearance. In particular, a known
        # process name executing from a new path (for example /tmp/ls) is the
        # exact executable-path signal C1 is meant to preserve. New baseline
        # variants belong in an explicitly reviewed retraining window.
        if any(self.counts[s.context].get(s.key, 0) == 0 for s in score.signals):
            return False
        if not (mode == "light" and confidence_before <= confidence_limit
                and score.evidence <= self.calm_rarity):
            return False
        # Keep the expected baseline in step with the expanding calm profile;
        # otherwise count growth silently makes the detector less sensitive.
        return self._observe_score(score)

    def freeze(self) -> None:
        """End the baseline window. From here on, rare things stay rare."""
        self.learning = False
        if self._baseline_samples:
            ordered = sorted(self._baseline_samples)
            self.calm_rarity = ordered[round(0.95 * (len(ordered) - 1))]
        self._baseline_samples = []


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
    # Exited while deep. Kept until dwell and decay release it, so the probes
    # stay attached across a burst of short-lived processes instead of paying
    # a ~370 ms detach on every exit. Its kernel map entries are already gone.
    exited: bool = False


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
                 backend: Optional[TracingBackend] = None,
                 audit: Optional[AuditLog] = None,
                 observe: bool = False,
                 score_log=None,
                 zones: Optional[Dict[str, list[str]]] = None) -> None:
        self.bus, self.source, self.host_id = bus, source, host_id
        # Calibration mode (proposal 3.5): score and log everything, switch
        # nothing, so the threshold can be chosen from a ROC sweep afterwards.
        self.observe = observe
        self.score_log = score_log
        # SimulatedBackend models the costs; EbpfBackend measures them. The
        # agent cannot tell the difference, which is the point.
        self.backend: TracingBackend = backend or SimulatedBackend()
        self.scorer = RarityScorer(zones=zones)
        self.ctl = controller or ModeController()
        # FR7: every transition and enforcement decision. In memory unless a
        # path is configured, so tests and the demo still produce the record.
        self.audit = audit or AuditLog(host_id=host_id)
        self.procs: Dict[int, ProcState] = {}
        self.comm_by_pid: Dict[int, str] = {}
        self.parent_by_pid: Dict[int, int] = {}
        self.denied: Dict[int, set] = defaultdict(set)   # LSM enforcement state
        # Enforcement is tracked per POLICY, not per pid: two overlapping leases
        # on one pid must not have the first expiry lift the second, and a
        # denial inherited by a child must be lifted with the policy.
        self.leases: Dict[str, dict] = {}
        self.pid_policies: Dict[int, set] = defaultdict(set)
        # Serialises transitions. Without it, a process promoted while the last
        # deep process is still being detached sees the old deep set, skips the
        # attach, and ends up "deep" with no probes attached.
        self._switch_lock = asyncio.Lock()
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
        bus.subscribe(Topic.ROLLBACK, self.on_rollback)

    # -- inbound from C3 ---------------------------------------------------
    async def on_external_alert(self, _topic: str, msg: ExternalAlert) -> None:
        if msg.host_id != self.host_id:
            return
        st = self.procs.get(msg.pid)
        if st is None:
            # Recorded, not silently dropped: an alert for a pid C1 never
            # reported means C3 and C1 disagree about the host's processes.
            self.audit.write("external_alert_ignored", pid=msg.pid,
                             alert_id=msg.alert_id, reason="pid not tracked by C1")
            return
        before = st.confidence
        st.confidence += msg.boost
        self.audit.write("external_alert", pid=msg.pid, alert_id=msg.alert_id,
                         boost=msg.boost, confidence_before=round(before, 3),
                         confidence_after=round(st.confidence, 3), reason=msg.reason)
        print(f"  [C1] E(t) from C3: pid {msg.pid} confidence "
              f"{before:.2f} -> {st.confidence:.2f}  ({msg.reason})")
        await self._maybe_switch(st, forced_trigger="external_alert")

    # -- inbound from C4 ---------------------------------------------------
    async def on_enforce(self, _topic: str, msg: EnforceHost) -> None:
        if msg.host_id != self.host_id:
            return
        policy = msg.policy_id or msg.msg_id
        if not getattr(self.backend, "enforcement_available", True):
            self._reject(policy, msg, "LSM enforcement not available on this host "
                                      "(bpf not in the active LSM list, or --lsm off)")
            return
        if msg.syscall not in SUPPORTED_DENIALS:
            self._reject(policy, msg, f"syscall {msg.syscall!r} not enforceable "
                                      f"(supported: {sorted(SUPPORTED_DENIALS)})")
            return
        # A pid in a LAAX message is only meaningful if C1 reported it (root
        # pid namespace; see HostEvent). Arming an unknown pid either silently
        # never matches or, worse, denies an unrelated process.
        if msg.pid not in self.procs:
            self._reject(policy, msg, "pid never reported by C1")
            return

        # Writes the pid into denied_pids, the map read by the
        # lsm/bprm_check_security program that returns -EPERM. laax_fork
        # extends it to children in-kernel from here on.
        latency = await self.backend.deny_add(msg.pid, msg.syscall)
        old = self.leases.get(policy)
        if old and old.get("timer"):
            old["timer"].cancel()                 # renewal: the new lease wins
        lease = old or {"pids": set(), "syscall": msg.syscall}
        lease["pids"].add(msg.pid)
        self.leases[policy] = lease
        self.pid_policies[msg.pid].add(policy)
        self.denied[msg.pid].add(msg.syscall)
        self.audit.write("enforce_armed", policy_id=policy, pid=msg.pid,
                         comm=self.procs[msg.pid].comm, syscall=msg.syscall,
                         lease_seconds=msg.lease_seconds,
                         map_write_ms=round(latency, 4))
        print(f"  [C1] LSM ARMED: deny {msg.syscall} for pid {msg.pid} and its "
              f"children ({latency:.3f}ms, lease {msg.lease_seconds:.0f}s, "
              f"policy {policy})")
        # The lease is C4's contract; C1 honours it locally so enforcement can
        # never outlive the policy that authorised it.
        lease["timer"] = asyncio.create_task(
            self._expire_denial(policy, msg.lease_seconds))

    def _reject(self, policy: str, msg: EnforceHost, reason: str) -> None:
        self.audit.write("enforce_rejected", policy_id=policy, pid=msg.pid,
                         syscall=msg.syscall, reason=reason)
        print(f"  [C1] LSM REFUSED: policy {policy} pid {msg.pid} -- {reason}")

    async def on_rollback(self, _topic: str, msg: Rollback) -> None:
        if msg.policy_id in self.leases:
            await self._release(msg.policy_id, msg.reason)

    async def _expire_denial(self, policy: str, after: float) -> None:
        await asyncio.sleep(after)
        await self._release(policy, "lease_expired")

    async def _release(self, policy: str, reason: str) -> None:
        lease = self.leases.pop(policy, None)
        if lease is None:
            return
        timer = lease.get("timer")
        if timer and timer is not asyncio.current_task():
            timer.cancel()
        released = []
        for pid in sorted(lease["pids"]):
            self.pid_policies[pid].discard(policy)
            if self.pid_policies[pid]:
                continue                          # another policy still holds it
            self.pid_policies.pop(pid, None)
            self.denied.pop(pid, None)
            await self.backend.deny_del(pid)
            released.append(pid)
        self.audit.write("enforce_released", policy_id=policy, reason=reason,
                         pids=released, syscall=lease["syscall"])
        print(f"  [C1] LSM RELEASED: {lease['syscall']} allowed again for "
              f"pid(s) {released} ({reason})")

    # -- process lifecycle (kernel source only) ----------------------------
    async def _on_fork(self, ev: HostEvent) -> None:
        """A deep or denied process forked. The kernel has already extended the
        gate and/or the denial to the child; record it so user space agrees."""
        flags = getattr(ev, "inherit_flags", 0)
        await self._retire_if_reused(ev.pid)
        parent = self.procs.get(ev.ppid)
        st = self.procs.get(ev.pid)
        if st is None:
            seed = parent.confidence * self.ctl.inherit if parent else 0.0
            st = ProcState(pid=ev.pid, comm=ev.comm, confidence=seed)
            self.procs[ev.pid] = st
        self.comm_by_pid[ev.pid] = ev.comm
        self.parent_by_pid[ev.pid] = ev.ppid
        if flags & INHERIT_DEEP and st.mode != "deep":
            await self._switch(st, "deep", "inherited")
        if flags & INHERIT_DENY:
            for policy in list(self.pid_policies.get(ev.ppid, ())):
                lease = self.leases.get(policy)
                if lease is None:
                    continue
                lease["pids"].add(ev.pid)
                self.pid_policies[ev.pid].add(policy)
                self.denied[ev.pid].add(lease["syscall"])
                self.audit.write("enforce_inherited", policy_id=policy,
                                 pid=ev.pid, parent_pid=ev.ppid)

    async def _on_exit(self, ev: HostEvent) -> None:
        """Forget the process. The kernel has already dropped its map entries,
        so a recycled pid starts with no confidence, mode or denial."""
        st = self.procs.get(ev.pid)
        if st is not None and st.mode == "deep":
            st.exited = True                      # decay_loop retires it
        else:
            self.procs.pop(ev.pid, None)
        self.comm_by_pid.pop(ev.pid, None)
        self.parent_by_pid.pop(ev.pid, None)
        for policy in self.pid_policies.pop(ev.pid, set()):
            lease = self.leases.get(policy)
            if lease:
                lease["pids"].discard(ev.pid)
        self.denied.pop(ev.pid, None)

    async def _retire_if_reused(self, pid: int) -> None:
        """The kernel recycled the pid of a process that exited while deep.
        Close the old entry's deep window before the new process is tracked,
        without touching the gate -- it may now belong to the new process."""
        old = self.procs.get(pid)
        if old is None or not old.exited:
            return
        if old.mode == "deep":
            await self._switch(old, "light", "pid_reused", touch_gate=False)
        self.procs.pop(pid, None)

    # -- mode switching ----------------------------------------------------
    async def _switch(self, st: ProcState, to_mode: str, trigger: str,
                      touch_gate: bool = True) -> None:
        async with self._switch_lock:
            if st.mode == to_mode:
                return                            # decided twice; already done
            await self._switch_locked(st, to_mode, trigger, touch_gate)

    async def _switch_locked(self, st: ProcState, to_mode: str, trigger: str,
                             touch_gate: bool = True) -> None:
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
            # An exited process's gate entry was deleted by laax_exit, and its
            # pid may already belong to someone else: leave the map alone.
            latency = await self.backend.gate_del(st.pid) if touch_gate else 0.0
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
        self.audit.write("mode_change", pid=st.pid, comm=st.comm,
                         from_mode=from_mode, to_mode=to_mode, trigger=trigger,
                         confidence=round(st.confidence, 3), operation=kind,
                         latency_ms=round(latency, 4),
                         latency_source="measured" if getattr(self.backend, "measured", False)
                         else "modelled")
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
        if self.observe:
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
            # Lifecycle events keep agent state in step with the kernel's maps.
            # They are not host telemetry, so they are not published to C3.
            if ev.kind == "fork":
                await self._on_fork(ev)
                continue
            if ev.kind == "exit":
                await self._on_exit(ev)
                continue
            self.comm_by_pid[ev.pid] = ev.comm
            # The kernel gives us the parent comm directly on exec (see
            # laax.bpf.c). Only fall back to a local pid->comm lookup when it
            # did not. Overwriting it unconditionally collapsed every lineage
            # key to ('?', child) and destroyed the rarity signal.
            if not getattr(ev, "comm_parent", None):
                ev.comm_parent = self.comm_by_pid.get(ev.ppid)  # type: ignore[attr-defined]
            # Prefer the kernel's grandparent name. Long-lived services often
            # started before C1, so our local process table is incomplete; an
            # unconditional assignment here used to erase gpcomm with None.
            grandparent_pid = self.parent_by_pid.get(ev.ppid)
            if not getattr(ev, "comm_grandparent", None) and grandparent_pid is not None:
                ev.comm_grandparent = self.comm_by_pid.get(grandparent_pid)  # type: ignore[attr-defined]
            self.parent_by_pid[ev.pid] = ev.ppid

            await self._retire_if_reused(ev.pid)
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

            # Only light-tier events are evidence. A deep kprobe event is extra
            # detail about an action the light tracepoints already reported
            # (finalize_exec duplicates sched_process_exec), or a per-send /
            # per-open record whose rate depends on what deep mode collects.
            # Scoring those too would let deep mode raise its own confidence
            # and hold itself open.
            if ev.hook in ("kprobe", "uretprobe"):
                rarity = 0.0
            else:
                confidence_before = st.confidence
                score = self.scorer.evaluate(ev)
                rarity = score.evidence
                learned = self.scorer.learn(
                    score, confidence_before=confidence_before, mode=st.mode,
                    confidence_limit=self.ctl.theta_down)
                self.ctl.update_confidence(st, rarity, self.scorer.baseline_rarity)
                if self.score_log is not None:
                    self.score_log.write(
                        ev, rarity, score.native_score, st.confidence,
                        self.scorer.learning, learned=learned,
                        explanation=score.explanation())

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
                # A payload, not a syscall name: every kernel event names its
                # syscall, so counting that reported every light event as a
                # suppressed payload.
                if ev.argv:
                    self.suppressed_deep += 1
                ev.argv, ev.syscall, ev.ret = None, None, None
            elif ev.argv:
                self.captured_deep += 1
                if ev.kind == "exec":
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
                    await self._switch(st, decision[0],
                                       "exit" if st.exited else "decay",
                                       touch_gate=not st.exited)
                    if st.exited:
                        self.procs.pop(st.pid, None)

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
            "lsm_policies_active": len(self.leases),
            "audit_records": len(self.audit.records),
            "latency_source": "measured" if getattr(self.backend, "measured", False) else "modelled",
        }


if __name__ == "__main__":
    import sys
    import laax_agent
    sys.exit(laax_agent.main())
