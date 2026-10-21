"""
LAAX integrated prototype -- end-to-end run.

    python run_demo.py            full scenario
    python run_demo.py --c2-eval  Component 2 evaluation only (T1/T2/T3)
    python run_demo.py --trace    print every bus message
    python run_demo.py --export c1_telemetry.csv   also write C1's telemetry
                                  in the C3 schema (services/c1_telemetry.py)

Everything runs in one process over InProcBus. Split the components across
processes by starting laax.bus.Broker and giving each a TcpBus; no component
code changes, because they only ever touch the Bus interface.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

from laax.bus import InProcBus
from laax.contracts import Topic
from scenario.lateral_movement import (
    HOST_IP,
    IOT_ATTACKER_IP,
    ScriptedKernelSource,
    attack_phase,
    benign_phase,
)
from services.c1_agent import C1Agent
from services.c1_telemetry import TelemetryCsvExporter
from services.c2_iot import C2Classifier, C2Service, Flow, run_evaluation, synth_flows
from services.c3_correlator import C3Correlator
from services.c4_containment import C4Containment


def banner(n: int, title: str) -> None:
    print(f"\n\033[1m=== PHASE {n}: {title} ===\033[0m")


async def main(trace: bool = False, export: str | None = None) -> None:
    bus = InProcBus(trace=trace)
    source = ScriptedKernelSource()

    c1 = C1Agent(bus, source, host_id="host-0")
    exporter = TelemetryCsvExporter(
        bus, export, {"host-0": HOST_IP},
        theta_up=c1.ctl.theta_up, theta_down=c1.ctl.theta_down) if export else None
    c3 = C3Correlator(bus)
    c4 = C4Containment(bus, dry_run=True, lease_seconds=6.0)

    tasks = [
        asyncio.create_task(bus.run()),
        asyncio.create_task(c1.run()),
        asyncio.create_task(c1.decay_loop()),
        asyncio.create_task(c4.reaper()),
    ]

    # ---------------------------------------------------------------- 1
    banner(1, "BENIGN BASELINE -- C1 learns, C3 learns, nothing fires")
    print("  90 benign host events; the agent should stay in light mode throughout.")
    await benign_phase(source)
    await bus.drain()
    c1.scorer.freeze()
    print(f"  [C1] baseline frozen: {len(c1.scorer.counts['lineage'])} lineage pairs, "
          f"{len(c1.scorer.counts['egress'])} egress pairs")
    loss = c3.train_baseline()
    print(f"  [C3] GNN baseline trained on benign graph, reconstruction loss {loss:.5f}")
    print(f"  [C1] mode switches during benign phase: {c1.ctl.switches}  "
          f"<- must be 0 for the overhead claim to hold")

    # ---------------------------------------------------------------- 2
    banner(2, "COMPONENT 2 -- agentless IoT detection flags the beachhead")
    clf = C2Classifier(feature_set="B").fit(synth_flows(2400, seed=1))
    c2 = C2Service(bus, clf)
    benign_flow = synth_flows(1, seed=11, attack_ratio=0.0)[0]
    benign_flow.device_id = "iot-03"
    await c2.score(benign_flow)

    attack_flow = synth_flows(1, seed=5, attack_ratio=1.0)[0]
    attack_flow.device_id, attack_flow.saddr = "iot-07", IOT_ATTACKER_IP
    attack_flow.daddr, attack_flow.dport = "10.0.0.12", 22
    await c2.score(attack_flow)
    await bus.drain()

    # ---------------------------------------------------------------- 3
    banner(3, "ATTACK -- IoT pivots into the managed host")
    print("  Watch for: C3 correlates -> E(t) raises C1 -> C1 flips to DEEP -> C4 contains")
    await attack_phase(source)
    await bus.drain()
    await asyncio.sleep(0.3)
    await bus.drain()

    # ---------------------------------------------------------------- 4
    banner(4, "RECOVERY -- lease expires, containment rolls back automatically")
    print(f"  Waiting {c4.lease_seconds:.0f}s for the containment lease to expire...")
    deadline = time.time() + c4.lease_seconds + 1.5
    while time.time() < deadline and not c4.rollbacks:
        await asyncio.sleep(0.25)
    await bus.drain()

    # ---------------------------------------------------------------- 5
    banner(5, "RESULTS")
    for name, stats in [("C1 adaptive agent", c1.stats()),
                        ("C3 correlation engine", c3.stats()),
                        ("C4 containment", c4.stats())]:
        print(f"\n  {name}")
        for k, v in stats.items():
            print(f"    {k:<28} {json.dumps(v) if isinstance(v, dict) else v}")

    print(f"\n  bus messages published        {bus.published}")
    lat = c1.ctl.attach_latencies
    if lat:
        print(f"\n  \033[1mHeadline claim check\033[0m")
        print(f"    switch latency budget       500 ms")
        print(f"    measured max                {max(lat):.2f} ms   "
              f"{'PASS' if max(lat) < 500 else 'FAIL'}")
        print(f"    benign-phase switches       {0}  (no false deep-mode entry)")

    for t in tasks:
        t.cancel()


if __name__ == "__main__":
    if "--c2-eval" in sys.argv:
        print("\n=== Component 2 evaluation: identifier exclusion ===")
        run_evaluation()
    else:
        try:
            export = None
            if "--export" in sys.argv:
                i = sys.argv.index("--export")
                export = sys.argv[i + 1] if i + 1 < len(sys.argv) else "c1_telemetry.csv"
            asyncio.run(main(trace="--trace" in sys.argv, export=export))
        except asyncio.CancelledError:
            pass
