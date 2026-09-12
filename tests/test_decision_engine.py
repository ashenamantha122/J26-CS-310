"""
test_decision_engine.py — run the Decision Engine against alerts.

Two modes:
  1. Built-in smoke tests (always run, no external file needed)
  2. Your real alert dataset (JSONL) — pass its path as argv[1]

Usage:
    python3 tests/test_decision_engine.py
    python3 tests/test_decision_engine.py /path/to/your/alerts.jsonl
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from decision_engine.engine import DecisionEngine  # noqa: E402

RULE_MATRIX = str(Path(__file__).resolve().parent.parent / "config" / "rule_matrix.yaml")

SMOKE_ALERTS = [
    # Low confidence port scan on a critical asset -> should stay NONE, not overreact
    {"alert_id": "smoke-1", "timestamp_ms": 0, "threat_type": "PORT_SCAN",
     "confidence": 0.2, "asset_id": "10.0.0.5", "asset_type": "IOT_AGENTLESS",
     "asset_criticality": "CRITICAL"},
    # High confidence lateral movement -> should ISOLATE
    {"alert_id": "smoke-2", "timestamp_ms": 0, "threat_type": "LATERAL_MOVEMENT",
     "confidence": 0.93, "asset_id": "host-07", "asset_type": "MANAGED_LINUX",
     "asset_criticality": "HIGH"},
    # Edge case: confidence exactly at a band boundary (0.5)
    {"alert_id": "smoke-3", "timestamp_ms": 0, "threat_type": "CREDENTIAL_ACCESS",
     "confidence": 0.5, "asset_id": "host-dc1", "asset_type": "MANAGED_LINUX",
     "asset_criticality": "CRITICAL"},
    # Edge case: unknown/malformed threat_type -> must hit default_rule, not crash
    {"alert_id": "smoke-4", "timestamp_ms": 0, "threat_type": "SOMETHING_NEW_C3_INVENTED",
     "confidence": 0.7, "asset_id": "host-02", "asset_type": "MANAGED_LINUX",
     "asset_criticality": "MEDIUM"},
    # Edge case: confidence out of [0,1] range (a bug upstream) -> must not crash
    {"alert_id": "smoke-5", "timestamp_ms": 0, "threat_type": "DATA_EXFILTRATION",
     "confidence": 1.4, "asset_id": "host-09", "asset_type": "MANAGED_LINUX",
     "asset_criticality": "HIGH"},
    # Edge case: missing field entirely -> decide() must still return, never raise
    {"alert_id": "smoke-6", "timestamp_ms": 0, "threat_type": "MALWARE_EXECUTION",
     "asset_id": "host-11", "asset_type": "MANAGED_LINUX", "asset_criticality": "HIGH"},
]


def run_alert_set(engine: DecisionEngine, alerts: list[dict], label: str):
    print(f"\n=== {label} ({len(alerts)} alerts) ===")
    level_counts: dict[str, int] = {}
    for alert in alerts:
        decision = engine.decide(alert)
        level_counts[decision.level] = level_counts.get(decision.level, 0) + 1
        print(f"  {alert.get('alert_id', '?'):10s} "
              f"{alert.get('threat_type', '?'):20s} "
              f"conf={alert.get('confidence', '?'):<5} "
              f"-> {decision.level:10s} ttl={decision.ttl_seconds:5d}s "
              f"[{decision.rule_matched}]")
    print(f"  -- level distribution: {level_counts}")


def main():
    engine = DecisionEngine(RULE_MATRIX)
    run_alert_set(engine, SMOKE_ALERTS, "Built-in smoke tests (including deliberately broken input)")

    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
        alerts = []
        with open(dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    alerts.append(json.loads(line))
        run_alert_set(engine, alerts, f"Real dataset: {dataset_path.name}")
    else:
        print("\n(Tip: pass your alert JSONL file as argv[1] to validate against the full dataset.)")


if __name__ == "__main__":
    main()
