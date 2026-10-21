"""
Per-event score log for threshold calibration (proposal section 3.5).
IT23184312 | Amantha M A

With `laax-agent run --observe --score-log FILE`, the agent scores every
light-tier event exactly as usual, writes one JSON line per event, and never
changes tracing mode. The ROC sweep (benchmarks/roc_eval.py) then replays any
threshold against the log offline, so one recording session yields the whole
curve instead of one operating point.

Line format (one object per scored event):
    {"ts": <epoch s>, "pid", "ppid", "comm", "parent", "exe", "kind",
     "rarity", "native_score", "confidence", "learning", "learned", "explanation",
     "scorer": "<name>"}
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional, TextIO


class ScoreLog:
    FLUSH_EVERY_S = 0.5

    def __init__(self, path: str, scorer: str = "v2") -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.path, self.scorer = path, scorer
        self._fh: Optional[TextIO] = open(path, "a", encoding="utf-8")
        self._last = 0.0
        self.records = 0

    def write(self, ev, rarity: float, native_score: float, confidence: float, learning: bool,
              *, learned: bool = False, explanation=None) -> None:
        if self._fh is None:
            return
        rec = {"ts": round(ev.ts, 6), "pid": ev.pid, "ppid": ev.ppid,
               "comm": ev.comm, "parent": getattr(ev, "comm_parent", None),
               "exe": ev.exe, "kind": ev.kind, "rarity": round(rarity, 4),
               "native_score": round(native_score, 4),
               "confidence": round(confidence, 4), "learning": learning,
               "learned": learned, "explanation": explanation or {},
               "scorer": self.scorer}
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.records += 1
        now = time.monotonic()
        if now - self._last >= self.FLUSH_EVERY_S:
            self._fh.flush()
            self._last = now

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
