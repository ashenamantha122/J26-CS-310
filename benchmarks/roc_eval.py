"""
Threshold calibration by ROC sweep (proposal section 3.5).

Inputs
    --scores FILE   score log(s) from `laax-agent run --observe --score-log FILE`
                    (repeat to compare scorers, e.g. v1 against v2)
    --labels FILE   attack windows from scenario/label_window.sh

    python3 benchmarks/roc_eval.py --scores v1.jsonl --labels labels.csv
    python3 benchmarks/roc_eval.py --scores v1.jsonl --scores v2.jsonl \
        --labels labels.csv --curve-out roc.csv

Unit of decision
    The agent's decision is "go deep", so the evaluation scores decisions,
    not events:
      positive  one labelled attack window. It is ACTIVATED at threshold t if
                any event in [start, end + grace] has confidence >= t.
      negative  one --bin-second slice of benign time (after the learning
                window, outside every attack window +- grace) that contains
                at least one event. Idle slices are left out: they cannot
                activate, and counting them would flatter the false
                activation rate.
    TPR = activated windows / windows   (activation recall)
    FPR = activated benign slices / benign slices
    plus false activations per benign hour, which is what an operator feels.

Operating point
    Recall first, as the proposal argues: a false activation costs seconds
    of deep tracing, a missed one costs the evidence. The recommended
    theta_up is the highest threshold whose recall meets --target-recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Window:
    start: float
    end: float
    label: str


def load_labels(path: str) -> List[Window]:
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(Window(float(row["start"]), float(row["end"]), row["label"]))
    if not out:
        raise SystemExit(f"{path}: no labelled windows")
    return sorted(out, key=lambda w: w.start)


def load_scores(path: str) -> Tuple[str, List[Tuple[float, float]]]:
    """(scorer name, [(ts, confidence)]) for events scored after learning."""
    name, events = None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            name = name or r.get("scorer") or path
            if r.get("learning"):
                continue
            events.append((float(r["ts"]), float(r["confidence"])))
    if not events:
        raise SystemExit(f"{path}: no scored events after the learning window")
    events.sort()
    return name or path, events


def units(events, windows: List[Window], grace: float, bin_s: float):
    """Max confidence per attack window, and per non-empty benign slice."""
    pos = []
    for w in windows:
        vals = [c for t, c in events if w.start <= t <= w.end + grace]
        pos.append(max(vals) if vals else 0.0)

    def in_attack(t: float) -> bool:
        return any(w.start - grace <= t <= w.end + grace for w in windows)

    slices: Dict[int, float] = {}
    for t, c in events:
        if in_attack(t):
            continue
        k = int(t // bin_s)
        slices[k] = max(slices.get(k, 0.0), c)
    benign_hours = len(slices) * bin_s / 3600.0
    return pos, list(slices.values()), benign_hours


def roc(pos: List[float], neg: List[float]):
    """Points (threshold, tpr, fpr) for every distinct threshold, plus AUC."""
    thresholds = sorted(set(pos) | set(neg) | {math.inf}, reverse=True)
    pts = []
    for t in thresholds:
        tpr = sum(p >= t for p in pos) / len(pos)
        fpr = sum(n >= t for n in neg) / len(neg) if neg else 0.0
        pts.append((t, tpr, fpr))
    pts.append((-math.inf, 1.0, 1.0))
    auc = 0.0
    for (_, y0, x0), (_, y1, x1) in zip(pts, pts[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return pts, auc


def operating_point(pts, target: float) -> Optional[Tuple[float, float, float]]:
    """Highest threshold with recall >= target (pts are high -> low)."""
    for t, tpr, fpr in pts:
        if tpr >= target and math.isfinite(t):
            return t, tpr, fpr
    return None


def evaluate(path: str, windows, args) -> dict:
    name, events = load_scores(path)
    pos, neg, hours = units(events, windows, args.grace, args.bin)
    pts, auc = roc(pos, neg)
    op = operating_point(pts, args.target_recall)
    at = None
    if args.theta is not None:
        tpr = sum(p >= args.theta for p in pos) / len(pos)
        fa = sum(n >= args.theta for n in neg)
        at = (args.theta, tpr, fa / len(neg) if neg else 0.0, fa / hours if hours else 0.0)
    return {"name": name, "path": path, "events": len(events), "windows": len(pos),
            "benign_slices": len(neg), "benign_hours": hours, "auc": auc,
            "points": pts, "op": op, "at_theta": at, "missed": [
                w.label for w, p in zip(windows, pos) if op and p < op[0]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROC sweep over observe-mode score logs")
    ap.add_argument("--scores", action="append", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--grace", type=float, default=2.0,
                    help="seconds after a window still credited to it (default 2)")
    ap.add_argument("--bin", type=float, default=10.0,
                    help="benign slice length in seconds (default 10)")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--theta", type=float, default=7.0,
                    help="also report this threshold (the agent's current theta_up)")
    ap.add_argument("--curve-out", help="write every ROC point to this CSV")
    args = ap.parse_args(argv)

    windows = load_labels(args.labels)
    results = [evaluate(p, windows, args) for p in args.scores]

    print(f"  attack windows: {len(windows)} ({', '.join(sorted({w.label for w in windows}))})")
    for r in results:
        print(f"\n  scorer {r['name']}  ({r['path']})")
        print(f"    scored events after learning   {r['events']}")
        print(f"    benign slices ({args.bin:.0f} s, non-empty) {r['benign_slices']}"
              f"  = {r['benign_hours']:.2f} h")
        print(f"    AUC                            {r['auc']:.3f}")
        if r["at_theta"]:
            t, tpr, fpr, perh = r["at_theta"]
            print(f"    at theta_up = {t:g}:  recall {tpr:.2f}, FPR {fpr:.3f}, "
                  f"{perh:.1f} false activations / benign hour")
        if r["op"]:
            t, tpr, fpr = r["op"]
            print(f"    recommended theta_up           {t:.3f}  (recall {tpr:.2f} >= "
                  f"{args.target_recall}, FPR {fpr:.3f})")
            if r["missed"]:
                print(f"    windows missed at that point   {', '.join(r['missed'])}")
        else:
            print(f"    no threshold reaches recall {args.target_recall}")

    if args.curve_out:
        with open(args.curve_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["scorer", "threshold", "tpr", "fpr"])
            for r in results:
                for t, tpr, fpr in r["points"]:
                    if math.isfinite(t):
                        w.writerow([r["name"], f"{t:.4f}", f"{tpr:.4f}", f"{fpr:.4f}"])
        print(f"\n  curve -> {args.curve_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
