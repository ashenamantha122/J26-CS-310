"""
ebpf_enforcer.py — user-space control plane for the eBPF/LSM path
(managed Linux endpoints). Same apply()/rollback() interface as
sdn_enforcer.py, same crash-isolation contract, so the supervisor and
Decision Engine treat both enforcers identically.

CRITICAL DESIGN CHOICE FOR YOUR TIMELINE:
eBPF/LSM is the riskiest, least-started subsystem (per your project
notes: "not yet started; requires kernel support verification"). This
module is written so that if the real BPF program isn't loaded yet —
or loading it fails on your kernel — the enforcer runs in SIMULATION
MODE: it validates inputs, maintains the rollback registry, writes the
audit log, and logs every action it *would* take, but doesn't touch a
real BPF map. This means:
  - You can wire the full Decision Engine -> eBPF path end-to-end and
    demo it TODAY, before hook_skeleton.bpf.c even compiles.
  - A kernel/BPF failure on demo day degrades to simulation mode and
    keeps the system running, instead of crashing Component 4 in front
    of the panel.
  - Swapping to real enforcement later is changing one flag
    (ebpf_enforcer.py's REAL_MODE) once hook_skeleton.bpf.c is loaded,
    not a rewrite.

Requires `bcc` (apt install bpfcc-tools python3-bpfcc) for REAL_MODE.
jsonschema/pyyaml-only dependencies otherwise — simulation mode has no
BPF dependency at all, so it runs anywhere, including this dev session.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("component4.ebpf_enforcer")

AUDIT_LOG = Path(__file__).resolve().parent.parent / "logs" / "ebpf_audit.log"

# Flip to True only after hook_skeleton.bpf.c is built and verified with
# check_ebpf_kernel.sh passing end-to-end on the target machine.
REAL_MODE = os.environ.get("COMPONENT4_EBPF_REAL", "0") == "1"


class EbpfEnforcer:
    def __init__(self):
        self._lock = threading.Lock()
        self._active: Dict[str, dict] = {}  # alert_id -> {pid/host, level, expires_at}
        # Same human-in-the-loop fix as sdn_enforcer.py: TTL expiry moves an
        # entry here instead of releasing it. It stays denied until a
        # reviewer calls confirm_release() or escalate().
        self._pending_review: Dict[str, dict] = {}
        self._stop_evt = threading.Event()
        self._ttl_thread = threading.Thread(target=self._ttl_watchdog, daemon=True)
        self._ttl_thread.start()
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

        self._bpf = None
        if REAL_MODE:
            self._init_real_bpf()
        else:
            logger.warning("ebpf_enforcer: running in SIMULATION MODE "
                            "(set COMPONENT4_EBPF_REAL=1 once hook_skeleton.bpf.c is verified)")

    def _init_real_bpf(self):
        try:
            from bcc import BPF  # imported lazily — not a hard dependency in sim mode
            self._bpf = BPF(src_file=str(Path(__file__).parent / "hook_skeleton.bpf.c"))
            logger.info("ebpf_enforcer: REAL BPF program loaded successfully")
        except Exception as e:  # noqa: BLE001
            logger.exception("ebpf_enforcer: failed to load real BPF program (%s) — "
                              "falling back to SIMULATION MODE for this session", e)
            self._bpf = None  # fall back rather than crash the whole process

    # ------------------------------------------------------------------
    # Public interface — mirrors SdnEnforcer exactly
    # ------------------------------------------------------------------

    def apply(self, decision) -> bool:
        try:
            return self._apply_unsafe(decision)
        except Exception as e:  # noqa: BLE001
            logger.exception("ebpf_enforcer: apply() failed for alert=%s target=%s: %s",
                              decision.alert_id, decision.target_id, e)
            self._audit("APPLY_FAILED", decision.alert_id, decision.target_id,
                        decision.level, error=str(e))
            return False

    def rollback(self, alert_id: str, reason: str = "manual") -> bool:
        """Direct, immediate release — bypasses pending-review. For an
        explicit administrative override, not called by the TTL watchdog."""
        try:
            return self._rollback_unsafe(alert_id, reason)
        except Exception as e:  # noqa: BLE001
            logger.exception("ebpf_enforcer: rollback() failed for alert=%s: %s", alert_id, e)
            self._audit("ROLLBACK_FAILED", alert_id, None, None, error=str(e))
            return False

    def confirm_release(self, alert_id: str, reviewer: str = "analyst") -> bool:
        """Human-in-the-loop release of a TTL-expired, pending-review entry.
        See sdn_enforcer.SdnEnforcer.confirm_release for the full rationale
        — same contract here: never raises, returns False if nothing was
        pending under that alert_id."""
        try:
            return self._confirm_release_unsafe(alert_id, reviewer)
        except Exception as e:  # noqa: BLE001
            logger.exception("ebpf_enforcer: confirm_release() failed for alert=%s: %s", alert_id, e)
            self._audit("CONFIRM_RELEASE_FAILED", alert_id, None, None, error=str(e))
            return False

    def escalate(self, alert_id: str, extend_seconds: int, reviewer: str = "analyst") -> bool:
        """Re-arm a pending-review entry for extend_seconds more instead of
        releasing it. The process/host was never actually un-denied."""
        try:
            return self._escalate_unsafe(alert_id, extend_seconds, reviewer)
        except Exception as e:  # noqa: BLE001
            logger.exception("ebpf_enforcer: escalate() failed for alert=%s: %s", alert_id, e)
            self._audit("ESCALATE_FAILED", alert_id, None, None, error=str(e))
            return False

    def active_containments(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._active)

    def pending_review(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._pending_review)

    def shutdown(self):
        self._stop_evt.set()
        self._ttl_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_unsafe(self, decision) -> bool:
        if decision.level == "NONE":
            self._audit("SKIP_NONE", decision.alert_id, decision.target_id, decision.level)
            return True

        pid_or_host = decision.target_id
        if self._bpf is not None:
            self._real_deny(pid_or_host)
            mode = "REAL"
        else:
            mode = "SIMULATED"

        expires_at = (time.time() + decision.ttl_seconds) if decision.ttl_seconds > 0 else None
        with self._lock:
            self._active[decision.alert_id] = {
                "target": pid_or_host,
                "level": decision.level,
                "expires_at": expires_at,
            }

        self._audit("APPLIED", decision.alert_id, pid_or_host, decision.level,
                    ttl=decision.ttl_seconds, mode=mode, rule_matched=decision.rule_matched)
        logger.info("ebpf_enforcer: [%s] APPLIED %s on %s (alert=%s, ttl=%ss)",
                    mode, decision.level, pid_or_host, decision.alert_id, decision.ttl_seconds)
        return True

    def _rollback_unsafe(self, alert_id: str, reason: str) -> bool:
        with self._lock:
            entry = self._active.pop(alert_id, None) or self._pending_review.pop(alert_id, None)
        if entry is None:
            return True
        if self._bpf is not None:
            self._real_allow(entry["target"])
        self._audit("ROLLED_BACK", alert_id, entry["target"], entry["level"], reason=reason)
        logger.info("ebpf_enforcer: ROLLED BACK %s on %s (alert=%s, reason=%s)",
                    entry["level"], entry["target"], alert_id, reason)
        return True

    def _confirm_release_unsafe(self, alert_id: str, reviewer: str) -> bool:
        with self._lock:
            entry = self._pending_review.pop(alert_id, None)
        if entry is None:
            return False
        if self._bpf is not None:
            self._real_allow(entry["target"])
        self._audit("RELEASED_CONFIRMED", alert_id, entry["target"], entry["level"], reviewer=reviewer)
        logger.info("ebpf_enforcer: RELEASED (confirmed by %s) %s on %s (alert=%s)",
                    reviewer, entry["level"], entry["target"], alert_id)
        return True

    def _escalate_unsafe(self, alert_id: str, extend_seconds: int, reviewer: str) -> bool:
        with self._lock:
            entry = self._pending_review.pop(alert_id, None)
            if entry is None:
                return False
            entry["expires_at"] = time.time() + extend_seconds
            self._active[alert_id] = entry
        self._audit("ESCALATED", alert_id, entry["target"], entry["level"],
                    reviewer=reviewer, extended_seconds=extend_seconds)
        logger.info("ebpf_enforcer: ESCALATED (by %s) %s on %s stays denied %ss more (alert=%s)",
                    reviewer, entry["level"], entry["target"], extend_seconds, alert_id)
        return True

    def _real_deny(self, pid_str: str):
        """Write PID=1 into the denied_pids BPF map. Only called when self._bpf is set."""
        pid = int(pid_str)
        denied_pids = self._bpf["denied_pids"]
        denied_pids[self._bpf.ctypes.c_uint(pid)] = self._bpf.ctypes.c_ubyte(1)

    def _real_allow(self, pid_str: str):
        pid = int(pid_str)
        denied_pids = self._bpf["denied_pids"]
        key = self._bpf.ctypes.c_uint(pid)
        try:
            del denied_pids[key]
        except KeyError:
            pass

    def _ttl_watchdog(self):
        """TTL expiry moves an entry to pending_review instead of releasing
        it automatically — see sdn_enforcer.py's _ttl_watchdog docstring
        for the full reasoning. The process/host stays denied until a
        human calls confirm_release() or escalate()."""
        while not self._stop_evt.wait(timeout=2.0):
            now = time.time()
            expired = []
            with self._lock:
                for alert_id, entry in self._active.items():
                    if entry["expires_at"] is not None and now >= entry["expires_at"]:
                        expired.append(alert_id)
                for alert_id in expired:
                    entry = self._active.pop(alert_id)
                    self._pending_review[alert_id] = entry
            for alert_id in expired:
                entry = self._pending_review[alert_id]
                self._audit("PENDING_REVIEW", alert_id, entry["target"], entry["level"],
                            note="TTL expired; stays denied until confirm_release() or escalate()")
                logger.warning("ebpf_enforcer: TTL expired for %s (alert=%s) — still DENIED, "
                                "awaiting confirm_release()/escalate()", entry["target"], alert_id)

    def _audit(self, event: str, alert_id: Optional[str], target: Optional[str],
               level: Optional[str], **extra):
        line = {"ts": time.time(), "event": event, "alert_id": alert_id,
                "target": target, "level": level, **extra}
        try:
            with open(AUDIT_LOG, "a") as f:
                f.write(json.dumps(line) + "\n")
        except OSError as e:  # noqa: BLE001
            logger.error("ebpf_enforcer: failed to write audit log (non-fatal): %s", e)
