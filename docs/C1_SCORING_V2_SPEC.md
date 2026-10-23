# C1 key function 1 — threat confidence scoring v2

## Purpose

Turn a light-tier host event into explainable evidence for C1's adaptive
tracing controller. The controller accumulates evidence with a one-sided
CUSUM and changes depth only at its calibrated hysteresis thresholds. A score
is never by itself an enforcement decision.

## Input → processing → output

| Stage | Definition |
|---|---|
| Input | A light `exec`, `connect`, or `accept` `HostEvent`; its process lineage, destination address/port, and (for exec) kernel-observed executable path. A C3 alert is separate cross-domain evidence. |
| Processing | Calculate rarity for each applicable signal against a Lidstone-smoothed (`alpha=0.5`) benign profile. Weight ancestry 0.50, zone egress 0.30, and executable path 0.20; add the weighted evidence to the process's one-sided CUSUM after subtracting the current calm-baseline mean. |
| Output | Raw evidence for the controller and the normalised `[0,1]` C3 telemetry score. When a score log is enabled, its explanation names every active signal, key, rarity, weight, contribution and reason; the agent does not allocate that record on the normal hot path. |

## Signals

1. **Two-step ancestry:** `(grandparent, parent, child)`. The kernel puts both
   parent and grandparent names into each exec event, so long-lived processes
   that started before C1 are covered. If either kernel field is unavailable,
   C1 uses its own process table; only then does it fall back to `(parent,
   child)` and record the fallback.
2. **Destination-zone egress:** `(process, destination zone, port)`. Zones
   are CIDR ranges in `deploy/laax-agent.conf`; longest-prefix wins. C1 labels
   addresses outside an explicit range as loopback, link-local, private,
   public, or unknown rather than guessing an IoT zone.
3. **Executable path:** `sched_process_exec` supplies the binary path. This
   protects against process-name spoofing and makes `/tmp` or `/dev/shm`
   execution visibly anomalous after the benign profile is built.
4. **C3 feedback:** C1 applies the configured `ExternalAlert.boost` delivered
   by C3 to the matching C1-reported pid. Boost calibration belongs to C3's
   owner; C1 does not reinterpret or scale another component's contract.

## Guarded baseline update

The initial baseline is a deliberately observed benign window. At freeze, C1
sets its calm-event limit to that window's 95th-percentile evidence and ties
the confidence guard to `theta_down` (the ROC-selected de-escalation point).
Afterwards it learns only from a **light-mode** event below both guards. Each
accepted calm event updates both profile counts and the baseline mean, avoiding
the slow loss of sensitivity caused by count drift. Deep or suspicious events
cannot teach the baseline. It also never learns a signal key that was absent
from the frozen profile: a familiar process name executing from `/tmp/ls`, for
example, must remain a new executable-path observation rather than being
absorbed on first use. Adding new benign keys requires an explicitly reviewed
baseline/retraining window.

## Process-name canonicalisation

Rarity keys are built from `comm`, but some comms carry a per-thread serial
that is unique to every thread in a pool: Firefox names its resolver threads
`DNS Res~er #648`, `#646`, … and rotates its socket workers. Keyed raw, every
such thread is a never-before-seen token scoring as maximally rare, so the
signal tracks "the browser spawned a new worker" rather than anything
adversarial. The ROC calibration of 22 Oct 2026 showed these dominated both the
attack windows and the benign false-positive wall (~13k events; benign
confidence peaking ~40,000), flattening v1 and v2 to an identical AUC ≈ 0.70.

`canon_comm()` collapses the volatile serial to the stable pool name before
keying, at the ancestry and egress sites. It is deliberately conservative — it
only strips a trailing ` #<n>` and folds the `kworker` pool, never touching
version suffixes or paths — so genuinely distinct programs stay distinct and
the adversarial signal is preserved. Offline re-score of the 22 Oct capture
(`benchmarks/rescore_canon.py`) estimates the benign false-positive rate at the
all-attacks-caught threshold falling from 0.333 to 0.024, with no loss of
recall. Confirm with a post-fix live run.

## Acceptance tests

`tests/test_c1_scoring_v2.py` verifies the agent's own grandparent lookup,
two-step and fallback ancestry, zone longest-prefix/IPv6/unknown-address
handling, executable-path evidence, preservation of kernel-supplied
grandparents when the local table is incomplete, the bounded native score, and
the guarded baseline policy (including a copied binary at a new path). Run
with:

```bash
python -m unittest tests.test_c1_scoring_v2 -v
```

ROC threshold selection remains an evidence task: collect labelled live attack
windows with `--observe --score-log`, then run `benchmarks/roc_eval.py`. The
threshold is not claimed to be calibrated until those labelled runs exist.
