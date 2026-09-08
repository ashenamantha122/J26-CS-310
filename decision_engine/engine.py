"""
engine.py — the Decision Engine subsystem of Component 4.

Converts a ThreatAlert (from Component 3, or your stub feed) into a
ContainmentDecision, using the explicit rule matrix in
config/rule_matrix.yaml. No ML here by design — see your proposal's
defensibility framing: a panel member can open the YAML and verify
every decision by hand.

Crash-isolation contract:
  - decide() NEVER raises. Any internal error returns a safe default
    decision (MONITOR) and logs the exception, because this function
    sits directly in the ZMQ receive callback path (see integration/
    resilient_link.py) and an uncaught exception there would otherwise
    only be caught by the generic handler — better to fail safe here,
    explicitly, with full context in the log.
  - enter_degraded_mode()/exit_degraded_mode() are called by the
    integration layer when Component 3 goes silent. In degraded mode
    the engine still processes whatever alerts it already has queued
    and still honours rollbacks/TTLs, it just stops expecting new
    alerts and surfaces that state for the dashboard/health check.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("component4.decision_engine")

CONFIDENCE_BANDS = {
    "LOW": (0.0, 0.5),
    "MEDIUM": (0.5, 0.8),
    "HIGH": (0.8, 1.0),
}


def confidence_to_band(confidence: float) -> str:
    for band, (lo, hi) in CONFIDENCE_BANDS.items():
        if band == "HIGH":
            if lo <= confidence <= hi:
                return band
        elif lo <= confidence < hi:
            return band
    return "LOW"  # defensive fallback for out-of-range input


@dataclass
class Decision:
    alert_id: str
    level: str          # NONE | MONITOR | RESTRICT | ISOLATE
    route_to: str        # MANAGED_LINUX | IOT_AGENTLESS
    target_id: str
    ttl_seconds: int
    rule_matched: str    # human-readable row identity, for the audit log


class DecisionEngine:
    def __init__(self, rule_matrix_path: str):
        self._path = Path(rule_matrix_path)
        self._lock = threading.Lock()
        self._rules: list[dict] = []
        self._default_rule: dict = {"level": "MONITOR", "ttl_seconds": 900}
        self._degraded = False
        self._load_rules()

    def _load_rules(self):
        try:
            with open(self._path, "r") as f:
                data = yaml.safe_load(f)
            with self._lock:
                self._rules = data.get("rules", [])
                self._default_rule = data.get("default_rule", self._default_rule)
            logger.info("decision_engine: loaded %d rules from %s", len(self._rules), self._path)
        except Exception as e:  # noqa: BLE001
            logger.exception("decision_engine: FAILED to load rule matrix (%s) — "
                              "falling back to default-only operation", e)
            with self._lock:
                self._rules = []

    def reload(self):
        """Hot-reload the rule matrix without restarting the process."""
        self._load_rules()

    def enter_degraded_mode(self):
        if not self._degraded:
            self._degraded = True
            logger.warning("decision_engine: entering DEGRADED mode (upstream alert source down) — "
                            "existing TTLs/rollbacks continue, new alerts not expected")

    def exit_degraded_mode(self):
        if self._degraded:
            self._degraded = False
            logger.info("decision_engine: upstream alert source recovered — exiting DEGRADED mode")

    @property
    def is_degraded(self) -> bool:
        return self._degraded

    def decide(self, alert: dict) -> Decision:
        """
        alert: dict matching THREAT_ALERT_SCHEMA in integration/resilient_link.py
        Never raises — worst case returns a MONITOR-level default decision.
        """
        try:
            return self._decide_unsafe(alert)
        except Exception as e:  # noqa: BLE001
            logger.exception("decision_engine: decide() failed for alert %s — "
                              "falling back to safe default (MONITOR): %s",
                              alert.get("alert_id", "<unknown>"), e)
            return Decision(
                alert_id=alert.get("alert_id", str(uuid.uuid4())),
                level="MONITOR",
                route_to=alert.get("asset_type", "ASSET_UNKNOWN"),
                target_id=alert.get("asset_id", "<unknown>"),
                ttl_seconds=900,
                rule_matched="ERROR_FALLBACK",
            )

    def _decide_unsafe(self, alert: dict) -> Decision:
        threat_type = alert["threat_type"]
        confidence = float(alert["confidence"])
        criticality = alert["asset_criticality"]
        band = confidence_to_band(confidence)

        with self._lock:
            rules = list(self._rules)
            default_rule = dict(self._default_rule)

        for idx, rule in enumerate(rules):
            if (rule.get("threat_type") == threat_type
                    and rule.get("confidence_band") == band
                    and criticality in rule.get("asset_criticality", [])):
                matched = f"rule[{idx}]:{threat_type}/{band}/{criticality}"
                logger.info("decision_engine: alert=%s matched %s -> %s",
                            alert["alert_id"], matched, rule["level"])
                return Decision(
                    alert_id=alert["alert_id"],
                    level=rule["level"],
                    route_to=alert["asset_type"],
                    target_id=alert["asset_id"],
                    ttl_seconds=int(rule.get("ttl_seconds", 0)),
                    rule_matched=matched,
                )

        logger.warning("decision_engine: alert=%s had NO matching rule "
                        "(threat_type=%s band=%s criticality=%s) — using default_rule",
                        alert["alert_id"], threat_type, band, criticality)
        return Decision(
            alert_id=alert["alert_id"],
            level=default_rule["level"],
            route_to=alert["asset_type"],
            target_id=alert["asset_id"],
            ttl_seconds=int(default_rule.get("ttl_seconds", 900)),
            rule_matched="default_rule",
        )
