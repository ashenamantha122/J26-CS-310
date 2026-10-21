"""
Component 1 forensic audit log (FR7).
IT23184312 | Amantha M A

Every mode transition and every enforcement decision -- armed, released,
rejected -- is written as one JSON object per line. This is the record an
analyst or an examiner reads to answer "what did the agent do, when, and why".

Tamper evidence
---------------
Each record carries `chain`, the SHA-256 of the previous record's chain value
concatenated with this record's canonical JSON. Editing, deleting or reordering
any line breaks every chain value after it, and `verify` reports the first
broken line. This detects tampering; it does not prevent it. An attacker with
root can rewrite the whole file consistently, so the chain head should also be
shipped off-host (the C3 export or the SOC console) to be trusted.

    python -m services.c1_audit verify /var/log/laax/audit.jsonl
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, TextIO, Tuple

GENESIS = "0" * 64


def _canonical(record: Dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _link(prev: str, record: Dict[str, Any]) -> str:
    return hashlib.sha256((prev + _canonical(record)).encode()).hexdigest()


class AuditLog:
    """Append-only, hash-chained JSON-lines log. path=None keeps it in memory."""

    def __init__(self, path: Optional[str] = None, host_id: str = "host-0") -> None:
        self.path, self.host_id = path, host_id
        self.records: List[Dict[str, Any]] = []
        self._prev = GENESIS
        self._fh: Optional[TextIO] = None
        if path:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            # Continue an existing chain rather than restarting it, so a
            # restart of the agent is not indistinguishable from a truncation.
            self._prev = _last_chain(path)
            self._fh = open(path, "a", encoding="utf-8")

    def write(self, event: str, **fields: Any) -> Dict[str, Any]:
        record = {"ts": round(time.time(), 6), "host_id": self.host_id,
                  "event": event, **fields}
        record["chain"] = _link(self._prev, record)
        self._prev = record["chain"]
        self.records.append(record)
        if self._fh is not None:
            self._fh.write(_canonical(record) + "\n")
            # Flushed per record: the log exists for the moment something goes
            # wrong, which is also the moment the agent may be killed.
            self._fh.flush()
        return record

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _last_chain(path: str) -> str:
    if not os.path.exists(path):
        return GENESIS
    last = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                last = line
    if last is None:
        return GENESIS
    return json.loads(last).get("chain", GENESIS)


def verify(path: str) -> Tuple[bool, int, str]:
    """(ok, records checked, message). Stops at the first broken link."""
    prev = GENESIS
    n = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return False, n, f"line {lineno}: not valid JSON"
            claimed = record.pop("chain", None)
            if claimed != _link(prev, record):
                return False, n, f"line {lineno}: chain broken (edited, removed or reordered)"
            prev = claimed
            n += 1
    return True, n, f"intact: {n} records, head {prev[:16]}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3 or sys.argv[1] != "verify":
        print("usage: python -m services.c1_audit verify <audit.jsonl>")
        raise SystemExit(2)
    ok, _n, msg = verify(sys.argv[2])
    print(msg)
    raise SystemExit(0 if ok else 1)
