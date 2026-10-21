# LAAX — integrated prototype

Lightweight Agent-Based & Agentless XDR Framework for Hybrid Enterprise-IoT
Environments. Project **J26-CS-310**.

One scripted attack runs through all four components and every arrow in the
system-overview diagram:

```
C2 scores IoT flows malicious
  -> C3 links the IoT address to a host process through a socket
  -> C3 raises CorrelationAlert (to C4) and E(t) (back to C1)
  -> C1 flips that process subtree to deep tracing; argv becomes visible
  -> C4 picks the minimum effective containment, applies SDN + eBPF LSM
  -> the lease expires and C4 rolls the containment back automatically
```

## Install the Component 1 agent (Ubuntu 24.04 / Debian 12+)

```bash
sudo bash ebpf/setup.sh                 # toolchain: clang, libbpf, bpftool
sudo apt install -y python3-zmq python3-protobuf dpkg-dev
bash packaging/build-deb.sh             # -> dist/laax-agent_<ver>_<arch>.deb
sudo apt install ./dist/laax-agent_*.deb

sudo laax-agent check --config /etc/laax/agent.conf   # what this kernel supports
sudo systemctl enable --now laax-agent
laax-agent status
journalctl -u laax-agent -f
```

Installed layout: `/usr/bin/laax-agent`, `/usr/lib/laax/` (agent + loader),
`/etc/laax/agent.conf`, `/var/log/laax/{telemetry.csv,audit.jsonl}`,
`/run/laax/status.json`. The service is not started on install: the operator
checks the kernel and the config first. `sudo bash packaging/test-install.sh`
runs the whole lifecycle (install, run, rotate, CURVE bus, purge) as a test.

**Production bus (protobuf over ZeroMQ, CURVE):**

```bash
sudo laax-agent keygen /etc/laax/curve --server          # broker host
sudo laax-agent keygen /etc/laax/curve --name agent      # each agent host
sudo mv /etc/laax/curve/agent.key /etc/laax/curve/clients/   # allow-list it on the broker
sudo systemctl enable --now laax-broker
# then in /etc/laax/agent.conf:  "bus": {"type": "zmq",
#   "curve_secret_key": "/etc/laax/curve/agent.key_secret",
#   "curve_server_key": "/etc/laax/curve/server.key"}
```

Requirement-by-requirement status and evidence: `docs/REQUIREMENTS_STATUS.md`.

**Threshold calibration (proposal 3.5).** Run the agent in observe mode, wrap
each Atomic Red Team test in a labelled window, then sweep:

```bash
sudo laax-agent run --mode ebpf --observe --score-log scores.jsonl --baseline 1800
bash scenario/label_window.sh labels.csv T1059.004 -- \
     pwsh -NoProfile -Command "Invoke-AtomicTest T1059.004 -TestNumbers 1"
python3 benchmarks/roc_eval.py --scores scores.jsonl --labels labels.csv --curve-out roc.csv
```

Observe mode scores and logs every event but never changes tracing depth, so
one session gives the whole ROC curve. Pass `--scores` twice to compare
scoring v1 with v2 on the same windows. Keep the benign period long: the
false-activation rate is measured per 10-second slice of benign activity.

## Run the four-component demo

No installs beyond what is already present (`networkx`, `scikit-learn`,
`torch`, `numpy`).

```bash
python run_demo.py            # the full end-to-end scenario
python run_demo.py --c2-eval  # Component 2 evaluation only (T1/T2/T3)
python run_demo.py --trace    # plus every bus message
```

## What is real and what is stubbed

| | Real | Stubbed |
|---|---|---|
| **C1** | rarity scorer with CUSUM confidence, hysteresis + dwell controller, in-kernel subtree inheritance, time-based decay, global-attach vs pid-gating split, LSM enforcement, audit log, protobuf/ZeroMQ export; runs on the live kernel and as an installed service | the kernel, on Windows only (`SimulatedBackend`); scoring v2 not yet written |
| **C2** | feature sets, threat-model exclusion, T1/T2/T3 evaluation, GBDT model | TON_IoT itself — flows are synthetic with the same structure |
| **C3** | multi-partite graph, GraphSAGE autoencoder, incident traversal, alert fan-out | trained on the demo's own benign window, not on a full dataset |
| **C4** | graduated levels, criticality policy, lease + rollback manager, dedup | Open vSwitch — the exact `ovs-ofctl` commands are printed; set `dry_run=False` to execute |

The stubs sit behind interfaces. Replacing one means editing one class, not the
components around it.

## Layout

```
laax/contracts.py     the canonical message types -- read this first
laax/bus.py           InProcBus (default) and TcpBus + Broker (dev, token auth)
laax/zbus.py          ZmqBus + ZmqBroker: protobuf over ZeroMQ, CURVE (production)
laax/pbcodec.py       contracts <-> protobuf (proto/laax.proto -> laax/laax_pb2.py)
laax/backend.py       TracingBackend: SimulatedBackend (modelled costs)
laax_agent.py         the agent CLI: run / check / status / keygen / verify-audit
services/c1_agent.py  adaptive eBPF agent
services/c1_audit.py  hash-chained audit log (FR7)
services/c1_preflight.py   kernel capability checks, LSM degrade
services/c1_telemetry.py   26-column C3 CSV export + conformance checker
packaging/            Debian package, systemd units, logrotate, install test
benchmarks/           kernel overhead, transition loss, transport throughput
services/c2_iot.py    agentless IoT flow classifier
services/c3_correlator.py   cross-domain correlation engine
services/c4_containment.py  hybrid containment controller
services/ebpf_source.py     EbpfBackend + EbpfKernelSource (measured costs)
scenario/             the lateral-movement scenario
ebpf/laax.bpf.c       kernel side: light tracepoints, deep kprobes, LSM hook
ebpf/loader.c         libbpf loader; owns the objects, times every operation
ebpf/Makefile         make / make check / make run
ebpf/setup.sh         one-shot Ubuntu toolchain setup
run_demo.py           the four-component scenario (any OS)
run_live.py           C1 against a real kernel (Linux, root)
docs/INTEGRATION.md   the seams, and three problems found while building this
docs/soc_console.html the analyst-facing console over C1..C4; open in a browser
```

## Running C1 on a real kernel

Verified on this machine: WSL2 kernel **6.18.33**, `/sys/kernel/btf/vmlinux`
present, `CONFIG_BPF_LSM=y`, `CONFIG_KPROBES=y`. CO-RE will work.

```powershell
wsl --install -d Ubuntu-24.04 --no-launch
```

```bash
wsl -d Ubuntu-24.04 -u root
cd "/mnt/e/Accepted RP/Prototype/ebpf"
bash setup.sh                       # toolchain, capability check, build

cd ..
python3 run_live.py --calibrate     # measure attach/detach latency
python3 run_live.py --seconds 60    # agent on live system activity
```

`setup.sh` handles the awkward part: WSL2 kernels are not in the apt archive,
so `linux-tools-$(uname -r)` will not resolve and `bpftool` has to be built
from source. `make check` reports whether the kernel can support the full
agent before you try.

### The enforcement hook

`CONFIG_BPF_LSM=y` is necessary but not sufficient — `bpf` must also be in the
*active* LSM list, which is a boot-time decision. In `%UserProfile%\.wslconfig`:

```ini
[wsl2]
kernelCommandLine = lsm=capability,landlock,yama,bpf
```

then `wsl --shutdown`. Confirm with `cat /sys/kernel/security/lsm`. Until then
run without `--lsm`; the loader disables autoload for that program so the rest
of the object still loads.

## Modelled numbers vs measured ones

`C1Agent.stats()` reports `latency_source: modelled | measured`. Never quote a
number without checking which one it says.

- **`SimulatedBackend`** (default, runs on Windows) is a *cost model* built
  from the medians below. Any number it produces is modelled, not measured.
- **`EbpfBackend`** times real `bpf_link` creation inside the loader with
  `CLOCK_MONOTONIC` and reports it back over the pipe.

Measured on WSL2 kernel 6.18.33.2, n=30 (`run_live.py --calibrate`), with the
four-probe deep tier of 21 Oct 2026 (`demo_evidence/calibration.txt`):

| operation | mean | median | p95 | max |
|---|---|---|---|---|
| `bpf_link` attach (deep, 4 probes) | 26.60 | 27.35 | 37.48 | 71.17 ms |
| `bpf_link` destroy (deep, 4 probes) | 376.32 | 371.24 | 445.16 | 447.20 ms |
| map update (gate a pid) | 0.009 | 0.009 | 0.012 | 0.013 ms |

Against the 500 ms budget: attach 7x headroom, **detach only 1.1x**. With the
two-probe tier of 16 Oct, detach was 175 ms; each kprobe costs ~95 ms to
unregister (measured one at a time), and the kernel serialises them, so detach
grows linearly with the number of deep probes. Destroying the links from
parallel threads did not overlap them. The fix on stock kernels is one
`kprobe_multi` (fprobe) link, but this WSL kernel is built without
`CONFIG_FPROBE`. Two things keep the detach off the critical path:

- leaving deep mode takes effect when the last pid leaves `deep_pids`
  (0.009 ms) -- the probes are no-ops from then on; teardown only reclaims
  their per-call cost;
- the loader tears the links down on a worker thread, so the ring buffer keeps
  being consumed during the ~370 ms instead of stalling at a transition.

A fifth deep probe would push the full teardown over 500 ms on this kernel.
Quote the worse direction, and say which kernel.

## Deep tier and process trees

Deep mode and LSM denials are inherited **in the kernel**, at fork
(`tp_btf/sched_process_fork`), before the child can run. Before 21 Oct this was
done only in user space, and the live kernel showed what that costs: a gated
shell's forked children exec'd with no deep event (`deep_payloads_captured 0`),
and a denied process could fork a child that exec'd freely. Both are now
checked on the live kernel by `scenario/test_inheritance.py`. Exit
(`sched_process_exit`, on the last thread) clears both maps so a recycled pid
starts clean.

Deep probes: `kprobe/finalize_exec` (full argv from the new image),
`kprobe/tcp_sendmsg`, `kretprobe/inet_csk_accept`, `kprobe/security_file_open`.

**Known reaction gap.** The score is updated when a process's exec event
reaches user space, so promotion always comes *after* that exec. A process's
own triggering exec is never captured deeply; only what it and its children do
afterwards is. In the 21 Oct live run (`demo_evidence/live_agent.txt`) all 209
deep events were file opens. The probe's short-lived commands had finished
before promotion, and the long-running shell scored below `theta_up` at birth.
Closing this gap is a job for the scoring engine (v2: ancestry, C3 pre-emptive
boost), not for the kernel side.

`SimulatedBackend` now carries these medians, so the Windows path models the
real asymmetry rather than inverting it.

## A caveat on benchmark numbers

WSL2 shares CPU with Windows and balloons memory, so it is fine for development
and for latency (which is dominated by kernel work, not scheduling) but noisy
for CPU and memory overhead. Take the final overhead figures on a
fixed-resource Hyper-V VM or bare metal, with frequency scaling disabled and
arms interleaved rather than run in blocks.
