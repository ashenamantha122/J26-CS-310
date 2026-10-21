# Component 1 — requirements traceability (21 Oct 2026)

Every objective and requirement of the proposal (*Proposal – Component 1*, §2
and Tables 5.1/5.2), what exists, and the evidence for it. **Status** is one of
Done, Partial, Not done, Deviation (deliberately changed from the proposal; must
be recorded with the supervisor).

All "measured" figures are from the live WSL2 kernel 6.18.33.2 on a 16-CPU
laptop. WSL2 shares the CPU with Windows; overhead figures there show order of
magnitude, and the thesis figures need the fixed-resource VM of proposal §3.5.

## Functional requirements (Table 5.1)

| ID | Requirement | Status | What exists | Evidence |
|---|---|---|---|---|
| FR1 | Monitor connections and process creation via XDP and tracepoints in light mode | **Deviation** (XDP) / Done (tracepoints) | Light tier: `sched_process_exec` (with real binary path), `sys_enter_connect` (IPv4 + IPv6, TCP/UDP from the socket), `sched_process_fork`, `sched_process_exit`. **No XDP**: XDP runs before any task exists and cannot attribute a packet to a process, which every C3 row needs (decision of 15 Oct). | `ebpf/laax.bpf.c`; `demo_evidence/live_agent.txt` |
| FR2 | Normalised threat confidence score from local events and C3 alerts | **Partial** | v1: information-theoretic rarity over a benign baseline (lineage, egress) with a one-sided CUSUM; C3 `ExternalAlert` boost. Normalised to [0, 1] at export. **Scoring v2 (Key Function 1) not written** — to be written by the student. | `services/c1_agent.py`; `tests/test_c1_agent_lifecycle.py` |
| FR3 | Attach deep kprobes above a threshold, detach below a lower one | **Done** | Hysteresis (θ↑ 7.0 / θ↓ 4.0), minimum dwell, time decay, per-pid gate, in-kernel inheritance at fork. | `tests/test_c1_hardening.py`; `demo_evidence/live_agent.txt` |
| FR4 | All transitions without restart, redeployment or intervention | **Done** | Deep programs verified at start-up, attached/detached as `bpf_link`s at runtime. | `demo_evidence/calibration.txt` |
| FR5 | Structured telemetry to C3 as Protocol Buffers over ZeroMQ | **Done** | `proto/laax.proto`, `laax/pbcodec.py`, `laax/zbus.py` (XSUB/XPUB broker, CURVE). The 26-column CSV for C3 training is kept alongside. | `tests/test_transport.py`; `demo_evidence/transport_throughput.txt`; `packaging/test-install.sh` |
| FR6 | Enforcement interface: deny a specified syscall for a specified process via eBPF/LSM | **Partial** | `lsm/bprm_check_security` denies **execve** for the process and every process it forks afterwards; per-policy leases, rollback, refusal of unknown pids. Other syscalls are refused, not faked. | `demo_evidence/lsm_test.txt`, `inheritance_test.txt`; `docs/INTEGRATION.md` |
| FR7 | Every mode transition and enforcement action as an auditable forensic record | **Done** | Hash-chained JSON-lines audit log; `laax-agent verify-audit` detects edits, deletions and reordering. | `services/c1_audit.py`; `demo_evidence/live_audit.jsonl` |

## Non-functional requirements (Table 5.2)

| ID | Requirement | Status | Measured | Evidence |
|---|---|---|---|---|
| NFR1 | < 2% average CPU under benign load; peak RAM within 10% of light | **Partial** — RAM met; CPU not demonstrated | **RAM:** peak total (agent + loader + BPF maps/programs) 48.3 MB light vs 49.6 MB deep, **+2.7%** (target ≤ 10%). **CPU:** kernel side per call: light exec 0.39 µs, connect 0.51 µs; deep file open 0.29 µs, deep exec (argv) 1.13 µs. User-space pipeline 40–80 µs per event (dominant). Idle loader 0.55% of one core. At 1,000 events/s the agent uses ~5% of **one** core; below 2% of a 2-vCPU host up to roughly 500 events/s. **Not run on the reference VM.** | `demo_evidence/kernel_overhead.txt`, `memory.txt` |
| NFR2 | Switch latency < 500 ms; enforcement denial < 1 ms | **Done, thin margin on detach** | Attach median 27 ms, max 71 ms. Detach median 371 ms, **max 447 ms (1.1× headroom)** with 4 probes; the light switch itself takes effect in 0.009 ms (gate removal). LSM decision 266 ns per exec in-kernel; arming a denial 0.007 ms (map write). | `demo_evidence/calibration.txt`, `kernel_overhead.txt` |
| NFR3 | < 1% event loss during transitions; no restarts | **Done** | 0 of 3,000 execs lost while deep mode was attached/detached under load; 0 ring-buffer drops in every run. | `demo_evidence/kernel_overhead.txt` part 3 |
| NFR4 | eBPF programs pass the verifier; telemetry channel authenticated | **Done** | All programs load through the verifier at start-up. ZeroMQ bus: CURVE encryption + key allow-list; an unlisted key and a plaintext client receive nothing. | `tests/test_transport.py`; `packaging/test-install.sh` |
| NFR5 | Thresholds and probe sets in one declarative config file | **Partial** | Thresholds, mode, LSM, bus, keys, paths in `/etc/laax/agent.conf` (strict: unknown keys rejected). The **probe set is fixed in the BPF object**, not configurable. | `tests/test_c1_hardening.py` ConfigTests |
| NFR6 | Export ≥ 10,000 events/s without backpressure loss | **Done (transport)** | 29,000 events/s end to end, CURVE on, 0 of 100,000 lost, 3 processes. The agent pipeline before the transport handles ~12,000–25,000 events/s per core (1 / per-event cost); not load-tested at that rate on a live kernel. | `demo_evidence/transport_throughput.txt` |

## Specific objectives (§2.2)

| # | Objective | Status |
|---|---|---|
| 1 | Light mode via XDP + tracepoints, < 1% CPU | Tracepoints done; XDP dropped (FR1); < 1% not demonstrated on the reference host |
| 2 | Confidence score fusing local signals + C3 alerts, in [0, 1] | v1 done; **v2 not written**. Calibration tooling ready: `--observe --score-log`, `scenario/label_window.sh`, `benchmarks/roc_eval.py` (AUC, recall-first θ↑, v1-vs-v2 comparison). **No ROC result yet** — needs Atomic Red Team runs |
| 3 | Kprobe attach/detach within 500 ms, no restart, < 1% loss | Done (NFR2, NFR3) |
| 4 | Deep tracing over memory, file and payload paths + export over Protobuf/ZeroMQ | File, exec argv, TCP send/accept done; **no memory (`mprotect`/`mmap` exec) probe, no uretprobes**; export done |
| 5 | Benchmark vs always-light, always-deep, Falco, Tetragon on TON_IoT replay | **Not done** (Phase 4). Light/deep/adaptive arms exist in `benchmarks/kernel_overhead.py`; no Falco/Tetragon arms, no TON_IoT, no reference VM |
| 6 | eBPF/LSM enforcement interface for C4, with latency | execve only (FR6); latency measured (NFR2) |

## Evaluation metrics (Table 3.1)

| Metric | Target | Status |
|---|---|---|
| Average CPU overhead | < 2% vs four baselines | Kernel-side measured; whole-agent ~5% of one core at 1,000 events/s; **no baselines, no reference host** |
| Peak RAM | within 10% of light | Met (+2.7%, `demo_evidence/memory.txt`) |
| Mode-switch latency | < 500 ms | Met (attach 71 ms max, detach 447 ms max) |
| Event loss in transitions | < 1% | Met (0%) |
| Telemetry coverage in attack windows | ≥ 90% of always-deep event types | **Not measured** (needs labelled attack windows) |
| Enforcement latency | sub-millisecond | Met (266 ns decision, 0.007 ms arming) |

## Product readiness

| Item | Status |
|---|---|
| Debian package (`packaging/build-deb.sh`) | Done — installs, starts, rotates logs, purges cleanly: 22/22 in `packaging/test-install.sh` |
| systemd unit, sandboxed and capability-bounded | Done — `systemd-analyze security`: 4.5 "OK"; runs as root with only CAP_BPF, CAP_PERFMON, CAP_SYS_ADMIN, CAP_SYS_RESOURCE |
| Preflight (`laax-agent check`) and graceful degrade without the BPF LSM | Done |
| Status (`laax-agent status`), SIGHUP log reopen, logrotate | Done |
| Key management (`laax-agent keygen`) | Done |
| Runs as a non-root user | Not done — the loader honours ambient capabilities, but the unit still runs as root |
| Kernel matrix (5.15 LTS, 6.x stock) | Not tested — only WSL2 6.18 |
| Container / pid-namespace awareness | Documented limitation: root-namespace pids only |

## Known limitations (state these, do not hide them)

1. **Reaction gap.** Promotion happens after the triggering exec reaches user
   space, so that exec is never captured deeply; only later activity is.
2. **Detach headroom** is 1.1× with four deep probes on this kernel (no
   `CONFIG_FPROBE` for a single `kprobe_multi` link).
3. **Light connect reads the sockaddr from user memory at syscall entry** —
   a racing thread can change it after the read (TOCTOU). The deep
   `tcp_sendmsg` record reads the kernel's socket and is not affected.
4. **Rarity keys on process names**, which a process can choose; the exec
   path is now captured for a v2 signal that cannot be chosen that way.
5. The user-space pipeline, not BPF, dominates CPU cost.
