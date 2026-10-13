"""
export_decisions.py — run the Decision Engine over a JSONL alert dataset
and write the full input+output of every decision to decisions.json,
for dashboard.html to read.

This does not replace tests/test_decision_engine.py — it's the same
engine, same rule_matrix.yaml, just exporting structured JSON instead of
printing a table, so you get a browsable GUI instead of terminal text.

Usage (run from your project root, e.g. ~/project/):
    python3 export_decisions.py path/to/your_alerts.jsonl

Writes: decisions.json (next to this script, or pass a second argv for
a custom output path).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# If you placed this file inside your component4 project folder, this finds
# decision_engine/ and config/ next to it. If you put it somewhere else,
# edit PROJECT_ROOT to point at the folder containing decision_engine/.
PROJECT_ROOT = ROOT.parent  # dashboard/ is a subfolder of the project root
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine.engine import DecisionEngine  # noqa: E402

RULE_MATRIX = str(PROJECT_ROOT / "config" / "rule_matrix.yaml")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 export_decisions.py path/to/alerts.jsonl [output.json]")
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "decisions.json"

    engine = DecisionEngine(RULE_MATRIX)

    results = []
    skipped = 0
    with open(dataset_path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                alert = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  line {line_no}: skipped, not valid JSON ({e})")
                skipped += 1
                continue

            decision = engine.decide(alert)
            results.append({
                "alert_id": alert.get("alert_id"),
                "threat_type": alert.get("threat_type"),
                "confidence": alert.get("confidence"),
                "asset_id": alert.get("asset_id"),
                "asset_type": alert.get("asset_type"),
                "asset_criticality": alert.get("asset_criticality"),
                "level": decision.level,
                "ttl_seconds": decision.ttl_seconds,
                "rule_matched": decision.rule_matched,
                "route_to": decision.route_to,
            })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} alerts ({skipped} skipped/malformed) -> {out_path}")
    print("Now open dashboard.html in a browser (see README.md for how to serve it).")


if __name__ == "__main__":
    main()
