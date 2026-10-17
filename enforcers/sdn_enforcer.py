"""
sdn_enforcer.py — wraps your existing, already-verified SDN baseline
(topo.py + containment_switch.py: Mininet + OS-Ken + OVS, pingall at
0% loss, file-watcher pushing high-priority OpenFlow drop rules) with:

  1. Fault isolation: a failure writing a rule is caught and reported
     back to the Decision Engine as a failed containment action — it
     never raises into the supervisor or kills the enforcer process.
  2. A rollback registry keyed by alert_id, so TTL expiry or a
     false-positive confirmation can cleanly remove exactly the rules
     one alert caused, without touching anyone else's.
  3. An audit log — every apply/rollback is one line, timestamped,
     alert_id-tagged — this is your "explainable policy records"
     novelty claim made real, and it's also what you paste into a
     SOC-style closure note for the panel demo.

TRANSPORT — matched to your real containment_switch.py:
Your verified controller's file-watcher thread polls a file
(default /tmp/containment_trigger) and expects plain-text commands,
one per line, in the form:

    ISOLATE <ip>
    RELEASE <ip>

It does NOT read a JSON rules file — so this module writes exactly
that text format. Commands are APPENDED (one per line), never
overwritten, because two containment decisions can arrive faster than
your watcher's poll interval; overwriting a single-value file would
silently drop whichever command lost the race. Your updated
containment_switch.py reads and processes every queued line, in
order, each time it polls, then clears the file — see the matching
_watch_trigger() change delivered alongside this file.

If your containment_switch.py's trigger file path differs from
/tmp/containment_trigger, change RULES_FILE below to match.

This module intentionally does NOT replace your verified switch
controller; it only writes commands to the same file your working
demo already watches, so your 0%-loss baseline stays untouched.

If your containment_switch.py instead exposes a function call or a
small HTTP/REST endpoint (OS-Ken's ofctl_rest app is the common
choice), swap _write_rule_entry()/_remove_rule_entry() for that call —
the public apply()/rollback() interface below does not need to change,
which is the point of having this wrapper layer at all.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("component4.sdn_enforcer")

# Must match the trigger file your containment_switch.py's _watch_trigger()
# thread polls. Your verified controller defaults to /tmp/containment_trigger.
RULES_FILE = Path("/tmp/containment_trigger")
AUDIT_LOG = Path(__file__).resolve().parent.parent / "logs" / "sdn_audit.log"


class SdnEnforcer:
    def __init__(self, rules_file: Path = RULES_FILE):
        self.rules_file = rules_file
        self._lock = threading.Lock()
        # alert_id -> {"host": ..., "level": ..., "expires_at": ..., "ip": ...}
        self._active: Dict[str, dict] = {}
        # alert_id -> same shape, but TTL has expired and release is now
        # waiting on a human decision (confirm_release / escalate) instead
        # of happening automatically. The host STAYS isolated while pending.
        self._pending_review: Dict[str, dict] = {}
        self._stop_evt = threading.Event()
        self._ttl_thread = threading.Thread(target=self._ttl_watchdog, daemon=True)
        self._ttl_thread.start()
        self.rules_file.parent.mkdir(parents=True, exist_ok=True)
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface — this is what the Decision Engine / supervisor calls
    # ------------------------------------------------------------------

    def apply(self, decision) -> bool:
        """
        decision: decision_engine.engine.Decision
        Returns True if the command was written successfully, False otherwise.
        NEVER raises — a failure here must not propagate.
        """
        try:
            return self._apply_unsafe(decision)
        except Exception as e:  # noqa: BLE001
            logger.exception("sdn_enforcer: apply() failed for alert=%s host=%s: %s",
                              decision.alert_id, decision.target_id, e)
            self._audit("APPLY_FAILED", decision.alert_id, decision.target_id,
                        decision.level, error=str(e))
            return False

    def rollback(self, alert_id: str, reason: str = "manual") -> bool:
        """Direct, immediate release — bypasses pending-review entirely.
        Used for an explicit administrative override (e.g. you've confirmed
        this was a false positive the moment it fired) — not called
        automatically by the TTL watchdog anymore. See confirm_release()
        for the normal human-in-the-loop path out of pending review."""
        try:
            return self._rollback_unsafe(alert_id, reason)
        except Exception as e:  # noqa: BLE001
            logger.exception("sdn_enforcer: rollback() failed for alert=%s: %s", alert_id, e)
            self._audit("ROLLBACK_FAILED", alert_id, None, None, error=str(e))
            return False

    def confirm_release(self, alert_id: str, reviewer: str = "analyst") -> bool:
        """
        Human-in-the-loop release: call this once a SOC analyst (or, in a
        demo, you) has actually looked at a TTL-expired containment and
        decided the host is clean. This is the ONLY normal path a host
        leaves containment once its TTL has expired — the watchdog no
        longer releases on its own, precisely because 'no further alerts
        arrived' is not proof the host is clean, only an absence of new
        evidence.

        Input:  alert_id of an entry currently in pending review, and who
                reviewed it.
        Processing: looks the entry up, writes the RELEASE command to the
                controller, logs who confirmed it and when.
        Output: True if a pending entry was found and released, False if
                there was nothing pending under that alert_id (never
                raises — same crash-isolation contract as apply/rollback).
        """
        try:
            return self._confirm_release_unsafe(alert_id, reviewer)
        except Exception as e:  # noqa: BLE001
            logger.exception("sdn_enforcer: confirm_release() failed for alert=%s: %s", alert_id, e)
            self._audit("CONFIRM_RELEASE_FAILED", alert_id, None, None, error=str(e))
            return False

    def escalate(self, alert_id: str, extend_seconds: int, reviewer: str = "analyst") -> bool:
        """
        The opposite call: a SOC analyst looked at a TTL-expired entry and
        decided it's still live — re-arm containment for extend_seconds
        more instead of releasing. The host was never actually released
        (pending review keeps it isolated), so this just resets its clock
        and moves it back into the active registry.

        Output: True if a pending entry was found and re-armed, False
        otherwise. Never raises.
        """
        try:
            return self._escalate_unsafe(alert_id, extend_seconds, reviewer)
        except Exception as e:  # noqa: BLE001
            logger.exception("sdn_enforcer: escalate() failed for alert=%s: %s", alert_id, e)
            self._audit("ESCALATE_FAILED", alert_id, None, None, error=str(e))
            return False

    def active_containments(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._active)

    def pending_review(self) -> Dict[str, dict]:
        """Entries whose TTL has expired and are still isolated, waiting on
        confirm_release() or escalate(). This is what a SOC dashboard
        should show as 'needs attention' — these are not errors, they're
        exactly the automation boundary described in the proposal."""
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
            return True  # correctly decided to do nothing — not a failure

        if decision.level == "MONITOR":
            # containment_switch.py only knows ISOLATE/RELEASE — there is no
            # network-level action for MONITOR. Log it as handled (watch-only)
            # rather than writing a command your watcher wouldn't understand.
            self._audit("MONITOR_NOOP", decision.alert_id, decision.target_id, decision.level,
                        note="MONITOR has no SDN-level action; visibility only")
            logger.info("sdn_enforcer: MONITOR for %s (alert=%s) — no OpenFlow action taken",
                        decision.target_id, decision.alert_id)
            return True

        ip = decision.target_id
        self._write_command("ISOLATE", ip)

        expires_at = (time.time() + decision.ttl_seconds) if decision.ttl_seconds > 0 else None
        with self._lock:
            self._active[decision.alert_id] = {
                "ip": ip,
                "level": decision.level,
                "expires_at": expires_at,
            }

        self._audit("APPLIED", decision.alert_id, decision.target_id, decision.level,
                    ttl=decision.ttl_seconds, rule_matched=decision.rule_matched)
        logger.info("sdn_enforcer: APPLIED %s on %s (alert=%s, ttl=%ss)",
                    decision.level, decision.target_id, decision.alert_id, decision.ttl_seconds)
        return True

    def _rollback_unsafe(self, alert_id: str, reason: str) -> bool:
        # Checks both registries: a direct rollback() call (e.g. an
        # immediate false-positive override) can target either an entry
        # that's still actively isolated or one already sitting in
        # pending review.
        with self._lock:
            entry = self._active.pop(alert_id, None) or self._pending_review.pop(alert_id, None)
        if entry is None:
            logger.debug("sdn_enforcer: rollback requested for unknown/already-cleared alert=%s", alert_id)
            return True
        self._write_command("RELEASE", entry["ip"])
        self._audit("ROLLED_BACK", alert_id, entry["ip"], entry["level"], reason=reason)
        logger.info("sdn_enforcer: ROLLED BACK %s on %s (alert=%s, reason=%s)",
                    entry["level"], entry["ip"], alert_id, reason)
        return True

    def _confirm_release_unsafe(self, alert_id: str, reviewer: str) -> bool:
        with self._lock:
            entry = self._pending_review.pop(alert_id, None)
        if entry is None:
            logger.debug("sdn_enforcer: confirm_release for alert=%s but nothing pending", alert_id)
            return False
        self._write_command("RELEASE", entry["ip"])
        self._audit("RELEASED_CONFIRMED", alert_id, entry["ip"], entry["level"], reviewer=reviewer)
        logger.info("sdn_enforcer: RELEASED (confirmed by %s) %s on %s (alert=%s)",
                    reviewer, entry["level"], entry["ip"], alert_id)
        return True

    def _escalate_unsafe(self, alert_id: str, extend_seconds: int, reviewer: str) -> bool:
        with self._lock:
            entry = self._pending_review.pop(alert_id, None)
            if entry is None:
                return False
            entry["expires_at"] = time.time() + extend_seconds
            self._active[alert_id] = entry
        self._audit("ESCALATED", alert_id, entry["ip"], entry["level"],
                    reviewer=reviewer, extended_seconds=extend_seconds)
        logger.info("sdn_enforcer: ESCALATED (by %s) %s on %s stays isolated %ss more (alert=%s)",
                    reviewer, entry["level"], entry["ip"], extend_seconds, alert_id)
        return True

    def _ttl_watchdog(self):
        """Background thread: when an active containment's TTL expires,
        it does NOT release automatically — it moves to pending_review and
        STAYS isolated. 'No new evidence arrived during the TTL window' is
        not the same as 'this host is confirmed clean', so release past
        this point requires an explicit confirm_release() call (or
        escalate() to keep it contained longer). This is the deliberate
        fix for the gap in pure auto-rollback: time-bounded automation
        without an unreviewed auto-release at the end of it."""
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
                self._audit("PENDING_REVIEW", alert_id, entry["ip"], entry["level"],
                            note="TTL expired; host remains isolated until confirm_release() or escalate()")
                logger.warning("sdn_enforcer: TTL expired for %s (alert=%s) — still ISOLATED, "
                                "awaiting confirm_release()/escalate()", entry["ip"], alert_id)

    # -- file I/O isolated into its own method so swapping transport is a one-place change --

    def _write_command(self, action: str, ip: str):
        """Append ONE plain-text command line matching containment_switch.py's
        updated _watch_trigger() parser: 'ISOLATE <ip>' or 'RELEASE <ip>'.
        Appending (not overwriting) so commands arriving faster than the
        watcher's poll interval queue up instead of clobbering each other."""
        with self._lock:
            with open(self.rules_file, "a") as f:
                f.write(f"{action} {ip}\n")
                f.flush()

    def _audit(self, event: str, alert_id: Optional[str], target: Optional[str],
               level: Optional[str], **extra):
        line = {
            "ts": time.time(),
            "event": event,
            "alert_id": alert_id,
            "target": target,
            "level": level,
            **extra,
        }
        try:
            with open(AUDIT_LOG, "a") as f:
                f.write(json.dumps(line) + "\n")
        except OSError as e:  # noqa: BLE001
            logger.error("sdn_enforcer: failed to write audit log (non-fatal): %s", e)
