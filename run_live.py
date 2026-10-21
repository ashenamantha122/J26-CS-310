"""
Component 1 against a real kernel. Linux only, needs root or CAP_BPF.

    sudo python3 run_live.py --calibrate      measure attach/detach latency
    sudo python3 run_live.py --seconds 60     run the agent on live system activity
    sudo python3 run_live.py --lsm            include the enforcement hook
    sudo python3 run_live.py --export c1_telemetry.csv
                                              write telemetry in the C3 schema

--calibrate is the one that matters for the proposal. It attaches and detaches
the deep probes N times against an already-loaded, already-verified object and
reports the distribution. That distribution IS the "switch latency under 500 ms"
claim; everything the simulated backend prints is a placeholder standing in for
this number until it exists.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

from services.ebpf_source import (
    EbpfBackend,
    EbpfKernelSource,
    EbpfLoader,
    LoaderUnavailable,
)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(int(len(xs) * p / 100.0), len(xs) - 1)]


async def calibrate(loader: EbpfLoader, n: int, settle: float) -> None:
    backend = EbpfBackend(loader)
    attach, detach, gate = [], [], []

    print(f"  warming up (one attach/detach pair, discarded)")
    await backend.attach_deep()
    await backend.detach_deep()

    for i in range(n):
        attach.append(await backend.attach_deep())
        gate.append(await backend.gate_add(1000 + i))
        await asyncio.sleep(settle)
        await backend.gate_del(1000 + i)
        detach.append(await backend.detach_deep())
        await asyncio.sleep(settle)
        print(f"\r  {i + 1}/{n}", end="", flush=True)
    print()

    print(f"\n  {'operation':<26}{'n':>5}{'mean':>10}{'median':>10}"
          f"{'p95':>10}{'max':>10}")
    for name, xs in (("bpf_link attach (deep)", attach),
                     ("bpf_link destroy (deep)", detach),
                     ("map update (gate a pid)", gate)):
        print(f"  {name:<26}{len(xs):>5}{statistics.mean(xs):>10.4f}"
              f"{statistics.median(xs):>10.4f}{pct(xs, 95):>10.4f}"
              f"{max(xs):>10.4f}   ms")

    # A mode SWITCH is whichever direction is being taken, so the budget has to
    # be judged against the worse of the two. Reporting attach alone would quote
    # the cheap half and flatter the result by an order of magnitude.
    worst_attach, worst_detach = max(attach), max(detach)
    worst = max(worst_attach, worst_detach)
    print(f"\n  switch latency budget      500.0000 ms")
    print(f"  worst attach (-> deep)     {worst_attach:>8.4f} ms   "
          f"{'PASS' if worst_attach < 500 else 'FAIL'}   "
          f"{500 / worst_attach:>5.1f}x headroom")
    print(f"  worst detach (-> light)    {worst_detach:>8.4f} ms   "
          f"{'PASS' if worst_detach < 500 else 'FAIL'}   "
          f"{500 / worst_detach:>5.1f}x headroom")
    print(f"  worst switch, either way   {worst:>8.4f} ms   "
          f"{'PASS' if worst < 500 else 'FAIL'}")

    ratio = statistics.mean(detach) / statistics.mean(attach)
    if ratio > 2:
        print(f"\n  NOTE: detach costs {ratio:.1f}x attach. Detaching a kprobe has to")
        print("  unregister it and wait out an RCU grace period, which attaching does")
        print("  not. Two consequences worth stating in the thesis:")
        print("    - min_dwell_s is not only an anti-flap device; leaving deep mode is")
        print("      the expensive direction, so dwell has a direct cost justification.")
        print("    - the honest headline is the worse direction, not the attach number.")

    print("\n  These are measurements. SimulatedBackend in laax/backend.py has been")
    print("  set from medians like these; re-run and update it if the kernel changes.")


async def live(loader: EbpfLoader, seconds: float, baseline: float,
               export: str | None = None, host_ip: str = "-",
               audit_path: str | None = None,
               ready_file: str | None = None) -> None:
    # Imported here, not at module scope: --calibrate needs only the loader, and
    # requiring networkx/torch to measure an attach latency would be absurd.
    from laax.bus import InProcBus
    from services.c1_agent import C1Agent
    from services.c1_audit import AuditLog

    bus = InProcBus()
    audit = AuditLog(audit_path)
    agent = C1Agent(bus, EbpfKernelSource(loader), backend=EbpfBackend(loader),
                    audit=audit)
    exporter = None
    if export:
        from services.c1_telemetry import TelemetryCsvExporter
        exporter = TelemetryCsvExporter(
            bus, export, {agent.host_id: host_ip},
            theta_up=agent.ctl.theta_up, theta_down=agent.ctl.theta_down)
        print(f"  exporting C3 telemetry to {export}")

    # C3 is optional here. Proving C1 works against a real kernel does not need
    # a correlation engine, and requiring torch on the test host to do it would
    # be a needless dependency.
    try:
        from services.c3_correlator import C3Correlator
        C3Correlator(bus)
        print("  C3 correlation engine attached")
    except ImportError as exc:
        print(f"  C3 not loaded ({exc.name} missing) -- running C1 only")

    tasks = [asyncio.create_task(bus.run()),
             asyncio.create_task(agent.run()),
             asyncio.create_task(agent.decay_loop())]

    print(f"  learning the baseline for {baseline:.0f}s -- run ordinary work now")
    if ready_file:
        # Tells a workload script the learning window has actually started.
        # Without it the script's clock starts at launch, and when start-up is
        # slow (importing torch for C3 takes seconds) its "anomalous" phase
        # lands inside the learning window, where switching is disabled.
        Path(ready_file).touch()
    await asyncio.sleep(baseline)
    agent.scorer.freeze()
    print(f"  baseline frozen: {len(agent.scorer.counts['lineage'])} lineage pairs, "
          f"{len(agent.scorer.counts['egress'])} egress pairs")
    print(f"  watching for {seconds:.0f}s -- try: bash -c 'nc -l 9999 &' or an "
          f"unusual process chain")
    await asyncio.sleep(seconds)

    for t in tasks:
        t.cancel()
    drops = await EbpfBackend(loader).dropped()
    stats = agent.stats()
    print("\n  Component 1 on a live kernel")
    for k, v in stats.items():
        print(f"    {k:<28} {v}")
    total = stats["events_emitted"] + drops
    print(f"    ring_buffer_drops            {drops}")
    if exporter:
        exporter.close()
        print(f"    c3_telemetry_rows            {exporter.rows}  -> {export}")
    print(f"    event_loss_pct               "
          f"{(100.0 * drops / total) if total else 0.0:.4f}  (budget: 1%)")
    audit.close()
    if audit_path:
        print(f"    audit_records                {len(audit.records)}  -> {audit_path}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--settle", type=float, default=0.05)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--baseline", type=float, default=30.0)
    ap.add_argument("--lsm", action="store_true")
    ap.add_argument("--export", metavar="CSV",
                    help="write C1 telemetry in the C3 schema to this file")
    ap.add_argument("--host-ip", default="-",
                    help="this host's address, for the host_ip column")
    ap.add_argument("--audit", metavar="JSONL",
                    help="append the hash-chained audit log (FR7) to this file")
    ap.add_argument("--ready-file", metavar="PATH",
                    help="touch this file when the baseline window starts "
                         "(scenario/live_probe.sh waits for it)")
    args = ap.parse_args()

    if not sys.platform.startswith("linux"):
        print("run_live.py needs Linux. On Windows use: wsl -d Ubuntu-24.04")
        return 2

    loader = EbpfLoader(lsm=args.lsm)
    try:
        await loader.start()
    except LoaderUnavailable as exc:
        print(f"cannot start loader: {exc}")
        return 1

    print(f"  loader ready (lsm={'on' if args.lsm else 'off'}); "
          f"light mode attached, deep mode loaded but NOT attached\n")
    try:
        if args.calibrate:
            await calibrate(loader, args.iterations, args.settle)
        else:
            await live(loader, args.seconds, args.baseline,
                       export=args.export, host_ip=args.host_ip,
                       audit_path=args.audit, ready_file=args.ready_file)
    finally:
        await loader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
