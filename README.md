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

## Run it

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
| **C1** | rarity scorer, EWMA confidence, hysteresis + dwell controller, subtree inheritance, time-based decay, global-attach vs pid-gating split, duty-cycle accounting | the kernel, on Windows only. `ebpf/laax.bpf.c` + `laax-loader` are written; run `run_live.py` on Linux for measured numbers |
| **C2** | feature sets, threat-model exclusion, T1/T2/T3 evaluation, GBDT model | TON_IoT itself — flows are synthetic with the same structure |
| **C3** | multi-partite graph, GraphSAGE autoencoder, incident traversal, alert fan-out | trained on the demo's own benign window, not on a full dataset |
| **C4** | graduated levels, criticality policy, lease + rollback manager, dedup | Open vSwitch — the exact `ovs-ofctl` commands are printed; set `dry_run=False` to execute |

The stubs sit behind interfaces. Replacing one means editing one class, not the
components around it.

## Layout

```
laax/contracts.py     the canonical message types -- read this first
laax/bus.py           InProcBus (default) and TcpBus + Broker
laax/backend.py       TracingBackend: SimulatedBackend (modelled costs)
services/c1_agent.py  adaptive eBPF agent
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

- **`SimulatedBackend`** (default, runs on Windows) is a *cost model*.
  `attach_ms = 3.2` is a placeholder of the right order of magnitude, nothing
  more. The `PASS` it prints against the 500 ms budget is a placeholder beating
  a placeholder.
- **`EbpfBackend`** times real `bpf_link` creation inside the loader with
  `CLOCK_MONOTONIC` and reports it back over the pipe.

Measured on WSL2 kernel 6.18.33.2, n=30 (`run_live.py --calibrate`):

| operation | mean | median | p95 | max |
|---|---|---|---|---|
| `bpf_link` attach (deep) | 12.38 | 13.92 | 16.50 | 17.60 ms |
| `bpf_link` destroy (deep) | 176.71 | 175.74 | 193.28 | 193.41 ms |
| map update (gate a pid) | 0.010 | 0.012 | 0.014 | 0.016 ms |

Against the 500 ms budget: attach 28x headroom, **detach 2.6x**. Detach is the
expensive direction because unregistering a kprobe waits out an RCU grace
period. Quote the worse of the two, not the attach figure.

`SimulatedBackend` now carries these medians, so the Windows path models the
real asymmetry rather than inverting it.

## A caveat on benchmark numbers

WSL2 shares CPU with Windows and balloons memory, so it is fine for development
and for latency (which is dominated by kernel work, not scheduling) but noisy
for CPU and memory overhead. Take the final overhead figures on a
fixed-resource Hyper-V VM or bare metal, with frequency scaling disabled and
arms interleaved rather than run in blocks.
