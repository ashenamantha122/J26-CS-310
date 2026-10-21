"""
Kernel-side cost of Component 1, measured on the live kernel. Linux, root.

    python3 benchmarks/kernel_overhead.py                 # all three parts
    python3 benchmarks/kernel_overhead.py --runs 5 --execs 2000

1. Workload slowdown. A workload process does three things the agent hooks --
   exec (posix_spawn /bin/true), TCP connect to a local listener, file open --
   and times each. Four configurations, interleaved and repeated:

     off           no agent loaded
     light         light tier attached (tracepoints only)
     deep-gated    deep kprobes attached, but this process NOT in deep_pids:
                   what every other process on the host pays while one
                   process is under investigation
     deep          deep kprobes attached and this process in deep_pids

2. Per-program cost: kernel.bpf_stats_enabled, then run_time_ns / run_cnt for
   every LAAX program (bpftool). bpf_stats adds its own accounting, so it is
   switched on only for this part.

3. Event loss during transitions: the workload execs a known number of
   processes while deep mode is attached and detached every 300 ms; every
   exec must produce a light event. Loss = execs without an event.

Honest scope: WSL2 shares the CPU with Windows and is noisy. These numbers
show order of magnitude and direction; the thesis figures need a
fixed-resource VM or bare metal, frequency scaling off (proposal 3.5).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Tuple
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.ebpf_source import EbpfBackend, EbpfLoader  # noqa: E402

WORKLOAD = r"""
import json, os, socket, subprocess, sys, time
subprocess.run([{marker!r}])                      # announces our root-ns pid
print("ready", flush=True)
cmd = json.loads(sys.stdin.readline())
out = {{}}
if "exec" in cmd:
    t = time.perf_counter()
    for _ in range(cmd["exec"]):
        pid = os.posix_spawn("/bin/true", ["/bin/true"], {{}})
        os.waitpid(pid, 0)
    out["exec"] = (time.perf_counter() - t) / cmd["exec"]
if "connect" in cmd:
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(512)
    addr = srv.getsockname()
    t = time.perf_counter()
    for _ in range(cmd["connect"]):
        c = socket.create_connection(addr); conn, _ = srv.accept()
        c.sendall(b"x"); conn.close(); c.close()
    out["connect"] = (time.perf_counter() - t) / cmd["connect"]
    srv.close()
if "open" in cmd:
    t = time.perf_counter()
    for _ in range(cmd["open"]):
        with open("/etc/hostname", "rb") as fh:
            fh.read(1)
    out["open"] = (time.perf_counter() - t) / cmd["open"]
print(json.dumps(out), flush=True)
"""


class Workload:
    """A workload process whose root-namespace pid we learn from the kernel."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    async def start(self, loader: EbpfLoader | None) -> int | None:
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", WORKLOAD.format(marker=self.marker),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        pid = None
        if loader:
            comm = os.path.basename(self.marker)[:15]
            while pid is None:
                ev = await asyncio.wait_for(loader.events.get(), 10)
                if ev.get("comm") == comm and ev.get("ppid"):
                    pid = int(ev["ppid"])
        await self.proc.stdout.readline()
        return pid

    async def run(self, spec: dict) -> dict:
        self.proc.stdin.write((json.dumps(spec) + "\n").encode())
        await self.proc.stdin.drain()
        result = json.loads(await self.proc.stdout.readline())
        await self.proc.wait()
        return result


async def drain_forever(loader: EbpfLoader, sink: list | None = None) -> None:
    while True:
        ev = await loader.events.get()
        if sink is not None:
            sink.append(ev)


async def one_run(config: str, spec: dict, marker: str) -> dict:
    if config == "off":
        w = Workload(marker)
        await w.start(None)
        return await w.run(spec)
    loader = EbpfLoader()
    await loader.start()
    be = EbpfBackend(loader)
    w = Workload(marker)
    pid = await w.start(loader)
    drain = asyncio.create_task(drain_forever(loader))
    if config in ("deep-gated", "deep"):
        await be.attach_deep()
    if config == "deep":
        await be.gate_add(pid)
    try:
        return await w.run(spec)
    finally:
        drain.cancel()
        await loader.stop()


def fmt_us(xs):
    return f"{statistics.mean(xs) * 1e6:8.2f} ± {statistics.pstdev(xs) * 1e6:6.2f} us"


async def part_slowdown(runs: int, spec: dict, marker: str) -> dict:
    configs = ["off", "light", "deep-gated", "deep"]
    results = {c: {k: [] for k in spec} for c in configs}
    for r in range(runs):
        order = configs[:]
        random.shuffle(order)                 # interleaved, so drift hits every arm
        for c in order:
            res = await one_run(c, spec, marker)
            for k, v in res.items():
                results[c][k].append(v)
        print(f"\r  run {r + 1}/{runs}", end="", flush=True)
    print()
    print(f"\n  {'per operation':<14}" + "".join(f"{c:>26}" for c in configs))
    for k in spec:
        print(f"  {k:<14}" + "".join(f"{fmt_us(results[c][k]):>26}" for c in configs))
    print(f"\n  {'overhead vs off':<14}" + "".join(f"{c:>26}" for c in configs))
    for k in spec:
        base = statistics.mean(results["off"][k])
        row = "".join(f"{100 * (statistics.mean(results[c][k]) - base) / base:>+25.1f}%"
                      for c in configs)
        print(f"  {k:<14}{row}")
    return results


def bpftool_progs() -> dict:
    out = subprocess.run(["bpftool", "prog", "show", "-j"], capture_output=True,
                         text=True, check=True).stdout
    return {p["name"]: p for p in json.loads(out) if p.get("name", "").startswith("laax_")}


async def part_prog_cost(spec: dict, marker: str) -> None:
    prev = Path("/proc/sys/kernel/bpf_stats_enabled").read_text().strip()
    Path("/proc/sys/kernel/bpf_stats_enabled").write_text("1")
    try:
        loader = EbpfLoader(lsm=True)
        await loader.start()
        be = EbpfBackend(loader)
        w = Workload(marker)
        pid = await w.start(loader)
        drain = asyncio.create_task(drain_forever(loader))
        await be.attach_deep()
        await be.gate_add(pid)
        await w.run(spec)
        progs = bpftool_progs()
        drain.cancel()
        await loader.stop()
    finally:
        Path("/proc/sys/kernel/bpf_stats_enabled").write_text(prev)
    print(f"\n  {'program':<18}{'calls':>10}{'mean ns/call':>15}{'total ms':>11}")
    for name in sorted(progs):
        p = progs[name]
        cnt, ns = p.get("run_cnt", 0), p.get("run_time_ns", 0)
        print(f"  {name:<18}{cnt:>10}{(ns / cnt if cnt else 0):>15.0f}{ns / 1e6:>11.2f}")
    print("  (workload process in deep mode, so every deep program fired for it)")


async def part_transition_loss(execs: int, marker: str) -> dict:
    loader = EbpfLoader()
    await loader.start()
    be = EbpfBackend(loader)
    w = Workload(marker)
    pid = await w.start(loader)
    seen: list = []
    drain = asyncio.create_task(drain_forever(loader, seen))
    await be.gate_add(pid)
    flips = 0

    async def toggle():
        nonlocal flips
        while True:
            await be.attach_deep()
            await asyncio.sleep(0.3)
            await be.detach_deep()
            await asyncio.sleep(0.3)
            flips += 2

    t = asyncio.create_task(toggle())
    await w.run({"exec": execs})
    await asyncio.sleep(1.0)                  # let the ring buffer drain
    t.cancel()
    drops = await be.dropped()
    drain.cancel()
    await loader.stop()
    light_execs = sum(1 for e in seen if e["kind"] == "exec" and e["depth"] == "light"
                      and e["comm"] == "true" and e["ppid"] == pid)
    deep_execs = sum(1 for e in seen if e["kind"] == "exec" and e["depth"] == "deep"
                     and e["ppid"] == pid)
    loss = 100.0 * (execs - light_execs) / execs
    print(f"\n  execs by the workload        {execs}")
    print(f"  mode transitions meanwhile   {flips}")
    print(f"  light exec events received   {light_execs}")
    print(f"  deep exec events (while attached) {deep_execs}")
    print(f"  ring-buffer drops            {drops}")
    print(f"  event loss                   {loss:.3f} %   "
          f"{'PASS' if loss < 1.0 else 'FAIL'} (budget < 1 %)")
    return {"execs": execs, "received": light_execs, "loss_pct": loss, "flips": flips}


def proc_cpu_s(pid: int) -> float:
    """utime + stime of a process, in seconds."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


async def part_delivery(execs: int, idle_s: float, marker: str) -> None:
    """Kernel-to-user-space delivery delay (the price of batched wake-ups),
    and what the loader costs while the host is idle (its 10 ms poll timer)."""
    import time
    loader = EbpfLoader()
    await loader.start()
    w = Workload(marker)
    pid = await w.start(loader)
    delays: list = []

    async def measure():
        while True:
            ev = await loader.events.get()
            if ev["kind"] == "exec" and ev["ppid"] == pid:
                # bpf_ktime_get_ns() and time.monotonic_ns() are both CLOCK_MONOTONIC.
                delays.append((time.monotonic_ns() - ev["ts_ns"]) / 1e6)

    task = asyncio.create_task(measure())
    await w.run({"exec": execs})
    await asyncio.sleep(0.5)
    task.cancel()
    loader_pid = loader.proc.pid
    drain = asyncio.create_task(drain_forever(loader))
    c0, t0 = proc_cpu_s(loader_pid), time.monotonic()
    await asyncio.sleep(idle_s)
    idle_pct = 100.0 * (proc_cpu_s(loader_pid) - c0) / (time.monotonic() - t0)
    drain.cancel()
    await loader.stop()
    delays.sort()
    q = lambda p: delays[min(int(len(delays) * p), len(delays) - 1)]
    print(f"\n  exec events measured          {len(delays)}")
    print(f"  delivery delay p50 / p95 / max   {q(0.5):.2f} / {q(0.95):.2f} / {delays[-1]:.2f} ms")
    print(f"  loader CPU while idle ({idle_s:.0f} s)   {idle_pct:.3f} % of one core")


async def part_pipeline(execs: int, marker: str) -> None:
    """User-space cost per event of the whole agent (loader + scoring + bus +
    C3 export), on real kernel events. This, not BPF, is the dominant cost."""
    import resource
    import time
    from laax.bus import InProcBus
    from services.c1_agent import C1Agent
    from services.c1_telemetry import TelemetryCsvExporter
    from services.ebpf_source import EbpfKernelSource

    loader = EbpfLoader()
    await loader.start()
    bus = InProcBus()
    agent = C1Agent(bus, EbpfKernelSource(loader), backend=EbpfBackend(loader))
    with tempfile.TemporaryDirectory() as d:
        exporter = TelemetryCsvExporter(bus, os.path.join(d, "t.csv"), {}, 7.0, 4.0)
        tasks = [asyncio.create_task(bus.run()), None]

        async def workload_run() -> None:
            # Workload.start() reads the loader queue itself, so the agent
            # consumes only once the workload has announced its pid.
            if tasks[1]:
                tasks[1].cancel()
            w = Workload(marker)
            await w.start(loader)
            tasks[1] = asyncio.create_task(agent.run())
            await asyncio.sleep(0.3)
            await w.run({"exec": execs})
            await asyncio.sleep(1.0)
            await bus.drain()

        # Steady state, not a cold start: learn the baseline from one run of
        # the workload, then measure an identical second run. Freezing an
        # empty baseline makes every event "rare" and measures mode-switch
        # churn (4,001 switches for 2,000 execs) instead of per-event cost.
        await workload_run()
        agent.scorer.freeze()
        if tasks[1]:
            tasks[1].cancel()
        w = Workload(marker)
        await w.start(loader)
        tasks[1] = asyncio.create_task(agent.run())
        await asyncio.sleep(0.3)
        e0, s0 = agent.emitted, agent.ctl.switches
        self0 = resource.getrusage(resource.RUSAGE_SELF)
        l0, t0 = proc_cpu_s(loader.proc.pid), time.monotonic()
        await w.run({"exec": execs})
        await asyncio.sleep(1.0)
        await bus.drain()
        self1 = resource.getrusage(resource.RUSAGE_SELF)
        l1, t1 = proc_cpu_s(loader.proc.pid), time.monotonic()
        events = agent.emitted - e0
        switches = agent.ctl.switches - s0
        for t in tasks:
            t.cancel()
        exporter.close()
    await loader.stop()
    py_s = (self1.ru_utime + self1.ru_stime) - (self0.ru_utime + self0.ru_stime)
    ld_s = l1 - l0
    print(f"\n  events through the agent      {events}  ({execs} execs, "
          f"{switches} mode switches, baseline learned first)")
    print(f"  python agent CPU              {py_s * 1e6 / max(events, 1):8.1f} us / event")
    print(f"  loader CPU (ring buffer+JSON) {ld_s * 1e6 / max(events, 1):8.1f} us / event")
    print(f"  total user space              {(py_s + ld_s) * 1e6 / max(events, 1):8.1f} us / event")
    print(f"  at the measured rate ({events / (t1 - t0):,.0f} events/s): "
          f"{100 * (py_s + ld_s) / (t1 - t0):.1f} % of one core")


def vm_hwm_kb(pid: int) -> int:
    """Peak resident set (VmHWM) of a process, in KiB."""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return 0


def bpf_memlock_kb() -> Tuple[int, int]:
    """Kernel memory charged to LAAX maps and programs (bpftool), KiB."""
    maps = json.loads(subprocess.run(["bpftool", "map", "show", "-j"], capture_output=True,
                                     text=True, check=True).stdout)
    progs = json.loads(subprocess.run(["bpftool", "prog", "show", "-j"], capture_output=True,
                                      text=True, check=True).stdout)
    laax_maps = {"events", "deep_pids", "denied_pids", "dropped", "lsm_stats", "laax_bpf.rodata"}
    m = sum(x.get("bytes_memlock", 0) for x in maps if x.get("name") in laax_maps)
    p = sum(x.get("bytes_memlock", 0) for x in progs if x.get("name", "").startswith("laax_"))
    return m // 1024, p // 1024


async def memory_run(mode: str, execs: int, marker: str) -> dict:
    """One configuration in this (fresh) process: the full agent, the workload
    in light mode or promoted to deep, peak RSS of agent and loader at the end."""
    from laax.bus import InProcBus
    from services.c1_agent import C1Agent, ProcState
    from services.ebpf_source import EbpfKernelSource

    loader = EbpfLoader()
    await loader.start()
    agent = C1Agent(InProcBus(), EbpfKernelSource(loader), backend=EbpfBackend(loader))
    bus_task = asyncio.create_task(agent.bus.run())
    w = Workload(marker)
    pid = await w.start(loader)
    agent.scorer.freeze()
    agent_task = asyncio.create_task(agent.run())
    if mode == "deep":
        st = ProcState(pid=pid, comm="python3", confidence=agent.ctl.theta_up)
        agent.procs[pid] = st
        await agent._switch(st, "deep", "benchmark")       # children inherit in-kernel
    await w.run({"exec": execs, "connect": execs, "open": execs * 10})
    await asyncio.sleep(1.0)
    maps_kb, progs_kb = bpf_memlock_kb()
    out = {"mode": mode, "agent_hwm_kb": vm_hwm_kb(os.getpid()),
           "loader_hwm_kb": vm_hwm_kb(loader.proc.pid), "bpf_maps_kb": maps_kb,
           "bpf_progs_kb": progs_kb, "events": agent.emitted}
    agent_task.cancel()
    bus_task.cancel()
    await loader.stop()
    return out


def part_memory(runs: int, execs: int) -> None:
    results = {"light": [], "deep": []}
    for _ in range(runs):
        for mode in ("light", "deep"):
            out = subprocess.run([sys.executable, __file__, "--part", "memory-run",
                                  "--mode", mode, "--execs", str(execs)],
                                 capture_output=True, text=True, timeout=300)
            line = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if not line:
                print(f"  {mode} run failed: {out.stderr[-300:]}")
                continue
            results[mode].append(json.loads(line[-1]))
    print(f"\n  {'(KiB, mean of ' + str(runs) + ')':<26}{'light':>12}{'deep':>12}{'deep vs light':>16}")
    for key, label in (("agent_hwm_kb", "agent peak RSS (Python)"),
                       ("loader_hwm_kb", "loader peak RSS"),
                       ("bpf_maps_kb", "BPF maps (kernel)"),
                       ("bpf_progs_kb", "BPF programs (kernel)")):
        lo = statistics.mean(r[key] for r in results["light"])
        de = statistics.mean(r[key] for r in results["deep"])
        print(f"  {label:<26}{lo:>12,.0f}{de:>12,.0f}{100 * (de - lo) / lo:>+15.1f}%")
    tot = lambda m: statistics.mean(r["agent_hwm_kb"] + r["loader_hwm_kb"] + r["bpf_maps_kb"]
                                    + r["bpf_progs_kb"] for r in results[m])
    lo, de = tot("light"), tot("deep")
    ev = lambda m: statistics.mean(r["events"] for r in results[m])
    print(f"  {'total':<26}{lo:>12,.0f}{de:>12,.0f}{100 * (de - lo) / lo:>+15.1f}%   "
          f"{'PASS' if de <= lo * 1.10 else 'FAIL'} (target: deep within 10% of light)")
    print(f"  events processed            {ev('light'):>10,.0f}{ev('deep'):>12,.0f}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--execs", type=int, default=1000)
    ap.add_argument("--connects", type=int, default=2000)
    ap.add_argument("--opens", type=int, default=20000)
    ap.add_argument("--loss-execs", type=int, default=3000)
    ap.add_argument("--idle", type=float, default=20.0, help="seconds of idle CPU sampling")
    ap.add_argument("--part", choices=["all", "slowdown", "progs", "loss", "delivery",
                                       "pipeline", "memory", "memory-run"], default="all")
    ap.add_argument("--mode", choices=["light", "deep"], default="light",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    if os.geteuid() != 0:
        print("needs root")
        return 2

    marker = f"/tmp/lxb{uuid.uuid4().hex[:6]}"
    shutil.copy("/bin/true", marker)
    spec = {"exec": args.execs, "connect": args.connects, "open": args.opens}
    try:
        if args.part == "memory-run":              # child process of part_memory
            print(json.dumps(await memory_run(args.mode, args.execs, marker)))
            return 0
        print(f"  kernel {os.uname().release}; {os.cpu_count()} CPUs")
        if args.part in ("all", "slowdown"):
            print(f"\n== 1. workload slowdown, {args.runs} interleaved runs per configuration")
            await part_slowdown(args.runs, spec, marker)
        if args.part in ("all", "progs"):
            print("\n== 2. per-program cost (kernel.bpf_stats_enabled)")
            await part_prog_cost(spec, marker)
        if args.part in ("all", "loss"):
            print("\n== 3. event loss during mode transitions")
            await part_transition_loss(args.loss_execs, marker)
        if args.part in ("all", "delivery"):
            print("\n== 4. delivery delay and idle cost")
            await part_delivery(args.execs, args.idle, marker)
        if args.part in ("all", "pipeline"):
            print("\n== 5. user-space pipeline cost per event")
            await part_pipeline(args.execs, marker)
        if args.part in ("all", "memory"):
            print(f"\n== 6. peak memory, light vs deep ({min(args.runs, 3)} fresh processes each)")
            part_memory(min(args.runs, 3), args.execs)
    finally:
        os.unlink(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
