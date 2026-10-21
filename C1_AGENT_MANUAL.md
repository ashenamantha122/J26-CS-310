# LAAX Component 1: Adaptive eBPF Endpoint Agent Manual
**Student:** Amantha M.A (IT23184312) | **Project:** J26-CS-310 (LAAX Framework)  
**Supervisor Alignment:** 60% Component Completion Milestone for PP1

---

## 1. Executive Summary & 60% Completion Breakdown

Component 1 has fulfilled **100% of the pre-PP1 core implementation (the 60% requirement)**. The agent is not a mock; it is a fully functioning, dual-mode autonomous endpoint security agent backed by real Linux eBPF kernel programs with dual-platform operation (Windows simulation + Linux kernel enforcement).

### The 60% Delivered Today (Pre-PP1)
1. **eBPF Kernel Instrumentation (`ebpf/laax.bpf.c`, `ebpf/loader.c`)**:
   - Light-mode tracepoints (`sched_process_exec`, `sys_enter_connect`) compiled via clang/CO-RE.
   - Deep-mode kprobes (`security_bprm_check`, `tcp_sendmsg`) attached on-demand via `bpf_link`.
   - eBPF LSM surgical containment hook (`lsm/bprm_check_security`) denying `execve` per-PID.
2. **Autonomous Agent Logic (`services/c1_agent.py`)**:
   - Information-theoretic `RarityScorer` (Laplace-smoothed n-gram rarity).
   - One-sided CUSUM change detector (pivoted from EWMA to detect APT slow-burn attacks).
   - `ModeController` with hysteresis ($\theta_{up}=7.0, \theta_{down}=4.0$) and minimum dwell floor ($3.0s$).
   - Process tree inheritance (suspicion and tracing depth flow down `ppid -> pid`).
   - Clock-driven exponential decay loop preventing duty-cycle ratchet.
3. **Telemetry & Contract Compliance (`services/c1_telemetry.py`)**:
   - Strict 26-column CSV telemetry exporter matching Component 3's agreed schema.
   - Conformance validator with zero schema violations.
4. **Autonomous Daemon & CLI (`laax_agent.py`)**:
   - Standalone daemon with automatic backend detection (eBPF vs Simulated).
   - Interactive viva demonstration console (`--interactive`).
   - In-process or distributed TCP broker support (`--bus tcp`).
5. **Overhead Profiling Suite (`benchmark_overhead.py`)**:
   - Empirical latency, memory, and throughput benchmark across Light, Deep, and Adaptive regimes.
6. **Empirical Evidence Pack (`demo_evidence/`)**:
   - Real kernel switch latency measurements on WSL2 Ubuntu kernel 6.18.33.2 (worst-case 190ms vs 500ms budget = 2.6x headroom).
   - 21 automated unit and lifecycle tests (100% pass rate).

### The Remaining 40% Roadmap (Post-PP1)
- **Phase 1 (Nov 2026)**: Integration with teammates' final, non-stub implementations of C2 (IoT classifier), C3 (GNN correlator), and C4 (SDN containment).
- **Phase 2 (Nov 2026)**: System-wide comparative performance benchmarking against Falco and Tetragon.
- **Phase 3 (Dec 2026)**: Evaluation on public benchmark datasets (DARPA OpTC / CICIDS-2017) for detection accuracy metrics (Precision, Recall, F1).
- **Phase 4 (Jan-Feb 2027)**: Final thesis writing and viva defense.

---

## 2. Quick Execution Guide

### Option A: Interactive Live Demonstration Mode (Recommended for Viva / Supervisor Meeting)
Allows you to press keys live to inject events, show the mode switch, demonstrate the RCU grace period latency, and trigger surgical containment:

```powershell
# From Windows PowerShell or Linux terminal:
cd "E:\Accepted RP\Prototype"
python laax_agent.py --interactive
```

**Hotkeys during interactive mode:**
- `[1]`: Spawns benign process (`git status`) $\rightarrow$ Agent stays in **LIGHT** mode.
- `[2]`: Spawns unusual lineage (`apache2 -> whoami`) $\rightarrow$ CUSUM score rises.
- `[3]`: Spawns reverse shell attack (`apache2 -> nc -e /bin/bash`) $\rightarrow$ Agent escalates to **DEEP** mode in milliseconds, captures full forensic payload!
- `[4]`: Injects C3 External Alert $E(t)$ $\rightarrow$ Boosts process confidence across domain boundary.
- `[5]`: Injects C4 LSM Execve Denial $\rightarrow$ Arms kernel map to block execution with timed auto-rollback.
- `[s]`: Prints live runtime statistics (duty cycle %, attach/detach latencies).
- `[q]`: Clean graceful shutdown with final summary report.

---

### Option B: Real Linux Kernel Execution (WSL2 / Ubuntu)
Executes directly against the Linux kernel using live eBPF tracepoints, kprobes, and BPF LSM:

```bash
# In WSL2 Ubuntu 24.04 as root:
cd "/mnt/e/Accepted RP/Prototype"
sudo python3 laax_agent.py --mode ebpf --lsm --export c1_telemetry.csv --audit c1_audit.jsonl

# or everything from the declarative config (CLI flags still override it):
sudo python3 laax_agent.py --config deploy/laax-agent.conf
```

`--mode ebpf` now **refuses** to run if the kernel side cannot start, instead of
quietly switching to synthetic events. `--allow-fallback` overrides that.

Audit log (FR7): every mode switch and every enforcement decision (armed,
inherited, released, refused) as hash-chained JSON lines. Check it with:

```bash
python3 -m services.c1_audit verify c1_audit.jsonl
```

Live-kernel proof of the fork/exit fixes (deep capture of children, LSM
covering forked children, exit cleanup):

```bash
python3 scenario/test_inheritance.py
```

---

### Option C: User-space processing benchmark (NOT an overhead measurement)
```powershell
python benchmark_overhead.py --events 3000 --iterations 3
```

This times the Python scoring pipeline on synthetic events with the
*simulated* backend. It does not measure kernel or system CPU overhead, so it
cannot support the < 2% objective, and its "data reduction" line counts
synthetic payloads. Do not present its numbers as agent overhead. The real
overhead benchmark (fixed-resource VM, always-light / always-deep / adaptive /
Falco / Tetragon, >= 10 runs) is Phase 4.

---

### Option D: Distributed Multi-Terminal TCP Bus Execution
To run C1, C2, C3, and C4 as independent processes on separate terminals or machines.
The broker requires a shared token so that no other local user can publish
enforcement requests:

**Terminal 1 (Message Broker):**
```powershell
$env:LAAX_BUS_TOKEN = "<a long random secret>"
python run_broker.py --port 5599
```

**Terminal 2 (Component 1 Agent):**
```powershell
$env:LAAX_BUS_TOKEN = "<the same secret>"
python laax_agent.py --bus tcp --broker 127.0.0.1:5599 --interactive
```

---

### Option E: Full Automated Scenario & Unit Tests

**Run full 4-phase end-to-end scenario:**
```powershell
python run_demo.py
```

**Run all 46 automated unit and lifecycle tests:**
```powershell
python -m unittest discover -s tests -v
```

---

## 3. Four Core Defense Points for the PP1 Panel

1. **Why not just leave Deep Mode on continuously?**
   - Deep probes fire on every exec, every TCP send, every accept and every file open of a gated process; in the 21 Oct live run one `ps` produced 144 deep file events. Always-deep pays that for every process all the time. How much CPU that costs is **not measured yet** -- that is the Phase 4 benchmark. Measured so far: in the 21 Oct live run the probes were attached for 32.8% of wall time (one attach, one detach) and 0 mode switches happened in the benign phase.
2. **Why does Detach take ~370 ms while Attach takes ~27 ms?**
   - Attaching creates `bpf_link`s against programs the verifier already accepted. Detaching a kprobe unregisters it and waits out an **RCU grace period**; the kernel serialises those (~95 ms per probe, 4 deep probes). This asymmetry justifies `min_dwell_s`, and it is why an exiting deep process is retired after its dwell rather than detaching at once (10 attach/detach cycles per burst became 1). Leaving deep mode takes effect in 0.009 ms when the gate entry is removed; the teardown runs in the background.
3. **Why did you pivot from EWMA to CUSUM?**
   - An EWMA applies exponential forgetting. A low-frequency, persistent anomaly (typical of an APT slow-burn attack) decays between events and never crosses the threshold. CUSUM accumulates deviations in excess of the benign baseline, reliably flagging persistence.
4. **Is eBPF LSM containment surgical?**
   - It is process-tree scoped and non-destructive. Returning `-EPERM` from `lsm/bprm_check_security` refuses `execve` for the target **and every process it forks afterwards** (extended in-kernel at fork). Nothing is signalled or killed; existing processes and their sockets keep running. Only `execve` can be denied today; other syscalls are refused, not faked. Until 21 Oct it was pid-only, which a fork bypassed.
