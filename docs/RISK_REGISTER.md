# Component 1 — risk register

Owner: IT23184312 Amantha M A · Prepared 21 Oct 2026 for PP1 · Review at every
supervisor meeting.

Likelihood (L) and impact (I): 1 low – 3 high. Score = L × I.
Status: **Open** (no mitigation yet), **Mitigating** (mitigation in place,
risk remains), **Closed** (eliminated, with evidence).

## Technical risks

| # | Risk | L | I | Score | Mitigation | Status | Evidence |
|---|---|---|---|---|---|---|---|
| T1 | Scoring v2 (Key Function 1) implemented but not calibrated: no labelled ROC run by PP1, so θ↑ is not yet evidence-based for v2 | 2 | 3 | 6 | v2 implemented and tested (81 unit tests; live two-step ancestry on 266/266 execs; unseen keys never learned); observe-mode ROC tooling ready (`benchmarks/roc_eval.py`); present ROC as the next step if Atomic Red Team runs are not done | Mitigating | `docs/C1_SCORING_V2_SPEC.md`, `tests/test_c1_scoring_v2.py` |
| T2 | Agent CPU cost exceeds the < 2% objective on busy hosts | 3 | 3 | 9 | Batched ring-buffer wake-ups (BPF 10.8 → 0.35 µs/call); per-event Python cost cut 73 → ~45 µs (v1). Scoring v2 measured 52–122 µs/event vs v1 47–60 (≈1.5×, noisy). Next: parse the ring buffer in-process instead of JSON; measure on the reference VM | Mitigating | `demo_evidence/kernel_overhead.txt` |
| T12 | Under scoring v2, C3's alert alone no longer promotes the entry process (demo: sshd 0.49 + 6.00 < θ↑ 7) | 3 | 2 | 6 | Stated honestly in the demo, README and console; resolve by ROC-selected θ↑ or a boost agreed with the C3 owner | Open | `run_demo.py` output |
| T3 | Detach latency breaches the 500 ms budget as deep probes are added (worst 447 ms with 4) | 2 | 2 | 4 | Light mode takes effect at gate removal (0.009 ms); teardown on a worker thread so collection never stalls; `kprobe_multi` on kernels with `CONFIG_FPROBE` | Mitigating | `demo_evidence/calibration.txt` |
| T4 | Reaction gap: the exec that triggers promotion is never captured deeply | 3 | 2 | 6 | In-kernel inheritance captures everything a promoted process does afterwards; closing the gap is a scoring v2 goal; stated as a limitation | Open | `demo_evidence/live_agent.txt` |
| T5 | False activations in benign use (deep mode too often) erode the overhead claim | 2 | 2 | 4 | Hysteresis + dwell; observe-mode ROC calibration measures false activations per benign hour before thresholds are fixed | Mitigating | `benchmarks/roc_eval.py` |
| T6 | Kernel compatibility: only tested on WSL2 6.18; BPF LSM absent on many distributions | 2 | 3 | 6 | `laax-agent check` preflight; telemetry-only degrade without the BPF LSM; test 5.15 LTS and stock 6.x before Phase 4 | Mitigating | `services/c1_preflight.py` |
| T7 | Evasion of light-mode telemetry (connect sockaddr TOCTOU at syscall entry; comm chosen by the process) | 2 | 2 | 4 | Deep `tcp_sendmsg` reads the kernel socket; exec path now captured for a v2 signal; documented | Mitigating | `docs/REQUIREMENTS_STATUS.md` |
| T8 | Pid-namespace mismatch: C4 targets a pid C1 never reported (containers) | 2 | 3 | 6 | Unknown pids refused and audited; contract documented in `docs/INTEGRATION.md` | Mitigating | `tests/test_c1_hardening.py` |
| T9 | Event loss under load | 1 | 3 | 3 | Ring-buffer drops counted; 0 of 3,000 lost during transitions; wake-up threshold well below buffer size | Closed (for tested load) | `demo_evidence/kernel_overhead.txt` part 3 |
| T10 | Forged commands on the bus (fake enforcement / alerts) | 1 | 3 | 3 | CURVE encryption + client key allow-list; dev TCP bus needs a token | Closed | `tests/test_transport.py`, `package_install.txt` |
| T11 | Benchmark numbers unreliable on WSL2 (shared CPU, ±25% spread) | 3 | 2 | 6 | Interleaved arms, ≥ 10 runs, mean ± sd; final figures on a fixed-resource VM with frequency scaling off (proposal 3.5) | Open | `demo_evidence/kernel_overhead.txt` |

## Project and integration risks

| # | Risk | L | I | Score | Mitigation | Status |
|---|---|---|---|---|---|---|
| P1 | C3 schema changes (argv, explanation, `ExternalAlert.confidence`) not agreed with the C3 member | 2 | 3 | 6 | Propose in writing; protobuf contract + codec test fails on drift; CSV kept for C3 training | Open |
| P2 | C4 needs denial of syscalls other than execve | 2 | 2 | 4 | Refused explicitly today (never faked); add LSM hooks as C4 specifies | Open |
| P3 | XDP removal not accepted by supervisors (proposal FR1 names XDP) | 2 | 2 | 4 | Rationale documented (no process attribution); get written sign-off at the next meeting | Open |
| P4 | Ownership: large parts of the prototype were AI-generated; panel questions on code the student cannot explain | 3 | 3 | 9 | Walkthroughs of every module; honest AI-use record; Key Function 1 written by the student | Open |
| P5 | Git history credibility (commit dates do not match when the work was done) | 3 | 3 | 9 | Discuss with the supervisor before PP1; keep all new history honest | Open |
| P6 | Demo fails live (WSL state, LSM not mounted, loader not built) | 2 | 3 | 6 | `laax-agent check` before the demo; banked evidence in `demo_evidence/`; cold-start rehearsal | Mitigating |
| P7 | Benchmark VM budget / time for the Phase 4 campaign | 2 | 2 | 4 | Budget line in the proposal (LKR 35,000); scripts ready (`benchmarks/`) so runs are cheap | Mitigating |

## Highest scores (act first)

T2 (CPU), P4 (ownership), P5 (git history), then T1, T4, T6, T8, T11, T12, P1, P6.
