#!/usr/bin/env python3
"""Offline re-score: measure the comm-canonicalisation fix on an existing capture.

Reads a score-log from a live ``--observe --score-log`` run and replays it
through a faithful model of the agent's scoring pipeline -- the rarity
*evidence* (same weights and smoothing as ``RarityScorer``) feeding the
per-pid CUSUM confidence, with the same baseline freeze and guarded learning
(an event above the learned ``calm`` p95 is never folded into the baseline).

It runs the identical reconstructed stream two ways, differing only in the
rarity key:

    raw   -- keys exactly as the live run saw them (reproduces the capture)
    canon -- volatile per-thread serials collapsed (the fixed scorer)

and reports, for each, the confidence threshold that still flags every attack
window and how many benign 10 s slices also fire there. This is the quantity
that blew up in the 22 Oct run (DNS-resolver threads reaching conf ~38000).
No kernel, no re-run: the capture is the dataset.

Caveat: models the dominant path (evidence -> CUSUM -> guarded learning). It
omits C3 E(t) boosts, inheritance seeding and time decay, which apply equally
to both key modes, so the raw-vs-canon comparison is unaffected.

    python3 benchmarks/rescore_canon.py --scores /root/cal/v2.jsonl \
        --labels /root/cal/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict

from services.c1_agent import canon_comm

WEIGHTS = {"ancestry": 0.50, "egress": 0.30, "executable": 0.20}
RAW_SCALE = 2.0
ALPHA, VOCAB = 0.5, 128


def ctx_of(name, key):
    if name == "ancestry":
        return "ancestry_2" if len(key) == 3 else "lineage"
    return name


def canon_key(name, ctx, key):
    key = tuple(key)
    if name == "ancestry" and ctx == "ancestry_2":
        return (canon_comm(key[0]), canon_comm(key[1]), canon_comm(key[2]))
    if name == "ancestry":
        return (canon_comm(key[0]), canon_comm(key[1]))
    if name == "egress":
        return (canon_comm(key[0]), key[1], key[2])
    return key


def load(path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        sigs = d.get("explanation", {}).get("signals", {})
        if not sigs:
            continue
        rows.append((d["ts"], d.get("pid", 0), bool(d.get("learning")), sigs))
    rows.sort(key=lambda r: r[0])
    return rows


def load_windows(path):
    return [(float(r["start"]) - 2.0, float(r["end"]) + 2.0, r["label"])
            for r in csv.DictReader(open(path))]


def run(rows, wins, canon):
    def attack_at(ts):
        return any(a <= ts <= b for a, b, _ in wins)

    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    def evidence_of(sigs):
        contribs = []
        for name, s in sigs.items():
            ctx = ctx_of(name, s["key"])
            key = canon_key(name, ctx, s["key"]) if canon else tuple(s["key"])
            contribs.append((name, ctx, key))
        ev = RAW_SCALE * sum(
            WEIGHTS[name]
            * -math.log2(max((counts[ctx].get(key, 0) + ALPHA)
                             / (totals[ctx] + ALPHA * VOCAB), 1e-12))
            for name, ctx, key in contribs)
        return ev, contribs

    # Phase 1: learning window -> baseline mean and calm p95; observe all keys.
    baseline_ev = []
    i = 0
    while i < len(rows) and rows[i][2]:            # while learning flag is True
        _, _, _, sigs = rows[i]
        ev, contribs = evidence_of(sigs)
        baseline_ev.append(ev)
        for _, ctx, key in contribs:
            counts[ctx][key] += 1
            totals[ctx] += 1
        i += 1
    baseline_rarity = sum(baseline_ev) / len(baseline_ev) if baseline_ev else 0.0
    calm = (sorted(baseline_ev)[round(0.95 * (len(baseline_ev) - 1))]
            if baseline_ev else 0.0)

    # Phase 2: post-freeze CUSUM with guarded learning.
    conf = defaultdict(float)
    out = []
    for ts, pid, _, sigs in rows[i:]:
        ev, contribs = evidence_of(sigs)
        conf[pid] = max(0.0, conf[pid] + (ev - baseline_rarity))
        attack = attack_at(ts)
        out.append((ts, conf[pid], attack))
        if not attack and ev <= calm:              # guarded learning
            for _, ctx, key in contribs:
                counts[ctx][key] += 1
                totals[ctx] += 1
    return out, calm


def wall(events, wins, bin_s=10.0):
    def which(ts):
        for idx, (a, b, _) in enumerate(wins):
            if a <= ts <= b:
                return idx
        return None
    window_max = defaultdict(float)
    benign_slice_max = defaultdict(float)
    for ts, conf, attack in events:
        idx = which(ts)
        if idx is not None:
            window_max[idx] = max(window_max[idx], conf)
        else:
            benign_slice_max[int(ts // bin_s)] = max(
                benign_slice_max[int(ts // bin_s)], conf)
    thr = min(window_max.values()) if window_max else 0.0
    fp = sum(1 for v in benign_slice_max.values() if v >= thr)
    peak_benign = max(benign_slice_max.values()) if benign_slice_max else 0.0
    return thr, len(window_max), fp, len(benign_slice_max), peak_benign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()
    rows = load(args.scores)
    wins = load_windows(args.labels)
    print(f"events: {len(rows)}   attack windows: {len(wins)}")
    for canon in (False, True):
        events, calm = run(rows, wins, canon)
        thr, caught, fp, tot, peak = wall(events, wins)
        tag = "canon (fixed) " if canon else "raw   (as-run)"
        fpr = fp / tot if tot else 0.0
        print(f"  {tag}: catch-all conf threshold={thr:10.1f}  "
              f"windows caught={caught}/{len(wins)}  "
              f"benign slices firing={fp}/{tot} (FPR {fpr:.3f})  "
              f"peak benign conf={peak:.0f}")


if __name__ == "__main__":
    main()
