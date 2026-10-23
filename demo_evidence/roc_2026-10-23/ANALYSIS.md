# ROC calibration run — 2026-10-22/23 (Kali 7.1.5, bare metal)

## Setup
- Host: Kali Linux, kernel 7.1.5+kali-amd64, 8 cores, 16 GB. BTF present, BPF-LSM active.
- v2 = `feature/c1-adaptive-agent` @63975b6; v1 = `c1-scoring-v1-reference` @1e69aa8.
- Both agents run in `--observe` (no blocking), 1800 s baseline, scores → `v2.jsonl` / `v1.jsonl`.
- Attacks: Atomic Red Team, 19 labelled windows over 6 tests
  (T1059.004-1, T1059.004-9, T1053.003-1, T1053.003-3, T1071.001-3, T1105-27),
  run by the user via `scenario/label_window.sh`. ~7 h non-empty benign activity.

## Result (target recall 0.9)
| | v1 | v2 |
|---|---|---|
| AUC | 0.698 | 0.699 |
| recall at FPR ≤ 0.20 | 0.053 | 0.053 |
| recall at FPR ≈ 0.324 | 1.00 | 0.895 |

The ROC is a step function: below ~32 % FPR only 1/19 attacks is caught.
v1 and v2 are effectively identical.

## Root cause (not a scoring-logic problem)
Top-scoring events *inside* attack windows are Firefox DNS-resolver threads, not the attacks:

```
T1053.003-3  conf=38838  connect / "DNS Res~er #648"
T1059.004-1  conf=38575  connect / "DNS Res~er #646"
T1105-27     conf=  122  connect / curl        <- the actual attack
T1053.003-1  conf=   67  exec / crontab        <- the actual attack
T1059.004-9  conf=   69  exec / id             <- the actual attack
```

The benign false-positive wall is the same: 13,465 `DNS Res~er #NNN` / `Socket Thread`
events scoring ≥ 20. Cause: `services/c1_agent.py` keys rarity on raw `ev.comm`
(line 195 egress, 182/185 ancestry). Browser DNS/socket threads carry a unique
`#<n>` suffix, so every one is a never-seen key → maximal rarity. The detector is
effectively keying on "browser doing DNS", which coincidentally overlaps ~32 % of
benign time and most attack windows.

## Fix (owner: student; scoring v2)
Canonicalise `comm` before keying — strip `" #<n>"`, collapse known thread-pool
names — at the ancestry and egress key sites. Expected effect: the DNS/socket
false positives collapse to one learned common key; attacks are then judged on
their real signals (new-zone `curl`, `crontab`, `base64→id`). Re-run after the fix;
only then is a v1-vs-v2 comparison meaningful.

## Fix applied (2026-10-23) and offline validation
`canon_comm()` added to `services/c1_agent.py` and applied at the ancestry and
egress key sites: a trailing ` #<n>` serial is stripped and the kworker pool is
folded, so a thread pool is one rarity key. Unit tests added in
`tests/test_c1_scoring_v2.py` (86 tests pass).

Validated offline on THIS capture with `benchmarks/rescore_canon.py` (replays
the recorded keys through the same evidence + per-pid CUSUM + guarded-learning
model, raw keys vs canon keys). The raw model reproduces the live run (peak
benign confidence 39,987 vs 38,838 observed; FPR 0.333 vs 0.324 from roc_eval),
which validates the model; the canon result is therefore trustworthy:

| at the confidence threshold catching all 19 windows | raw | canon |
|---|---|---|
| windows caught | 19/19 | 19/19 |
| benign false-positive rate | 0.333 | **0.024** |
| peak benign confidence | 39,987 | **191** |

→ ~14x fewer false activations, DNS-resolver wall removed, no loss of recall.
Result file: `rescore_canon.txt`. This is an offline estimate; confirm with a
clean live re-run after the fix (that run is also the fair v1-vs-v2 comparison).

AI-use note: `canon_comm` and `rescore_canon.py` were written with Claude Code
(Opus) assistance on 2026-10-23, reviewed by the student; record in the PP1
AI-use log.

## Post-fix LIVE re-run (2026-10-23, same Kali host)
v2 rebuilt from the pushed fix (fa994cb); v1 reference unchanged. 12 ART attacks
(all 6 tests), fresh baseline, logs `v2b.jsonl`/`v1b.jsonl`, labels `labels2.csv`,
`roc2_result.txt` / `roc2.csv`.

| target recall 0.9 | v1 (no fix) | v2 (fixed) |
|---|---|---|
| AUC | 0.420 | **0.992** |
| recommended θ↑ | 28.9 (FPR 0.655) | **45.7 (FPR 0.011)** |
| recall at θ↑ | 1.00 | 0.92 (misses T1053.003-3) |

The canonicalisation fix is confirmed live: v2 goes from tied-with-v1 (~0.70 AUC,
~0.32 FPR on 22 Oct) to AUC 0.992 at ~1% FPR, clearly separating from unfixed v1
(0.420) on the same stream. Corroborates the offline ~14x estimate.

Limitations (state honestly): benign window was short (0.49 h, 177 slices), so the
FPR figure is coarse; v2 still misses the cron-file-drop test (T1053.003-3) at the
0.9-recall threshold. A longer benign run would firm up the numbers for the thesis.

## Honest status for PP1
- The 22 Oct run is a valid *preliminary* calibration that surfaced a real
  feature-extraction bug. Do **not** claim v2 > v1 or any usable detection rate
  from that run; the v1-vs-v2 question is still open until a post-fix live re-run.
- The offline re-score shows the fix direction is strong but is an estimate, not
  a live measurement.
- Raw logs retained on the test host at `/root/cal/` (v2.jsonl ~98 MB, v1.jsonl ~34 MB);
  `roc.csv`, `roc_result.txt`, `labels.csv`, `rescore_canon.txt` copied here.
