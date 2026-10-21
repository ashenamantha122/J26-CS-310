"""
NFR6: can the C1 -> C3 export sustain 10,000 events/s without loss?

Three separate processes, as deployed: the broker (run_broker.py), a
subscriber playing C3, and this process publishing HostEvents as C1 does
(protobuf over ZeroMQ, CURVE on by default). Counts what C3 receives.

    python3 benchmarks/transport_throughput.py --events 100000
    python3 benchmarks/transport_throughput.py --events 100000 --no-curve

Measures the transport alone (encode, CURVE, proxy hop, decode) on one host.
The kernel ring buffer and the scoring pipeline are measured separately.
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from laax.contracts import HostEvent, Topic  # noqa: E402
from laax.zbus import CurveClient, ZmqBus, generate_keypair  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def subscriber(xsub, xpub, keyfiles, n, ready, result) -> None:
    async def go():
        curve = CurveClient.from_files(*keyfiles) if keyfiles else None
        bus = ZmqBus(xsub, xpub, curve=curve)
        count, first, last = 0, None, None
        done = asyncio.Event()

        async def on_event(_t, _m):
            nonlocal count, first, last
            count += 1
            last = time.time()
            if first is None:
                first = last
            if count >= n:
                done.set()

        bus.subscribe(Topic.HOST_EVENT, on_event)
        await bus.start()
        task = asyncio.create_task(bus.run())
        ready.set()
        try:
            await asyncio.wait_for(done.wait(), 120)
        except asyncio.TimeoutError:
            pass
        task.cancel()
        bus.close()
        result.put((count, first, last))
    asyncio.run(go())


async def publish(xsub, xpub, keyfiles, n) -> tuple:
    curve = CurveClient.from_files(*keyfiles) if keyfiles else None
    bus = ZmqBus(xsub, xpub, curve=curve)
    await bus.start()
    await asyncio.sleep(0.5)
    ev = HostEvent(host_id="host-0", pid=4471, ppid=1400, comm="bash",
                   exe="/usr/bin/bash", kind="exec", hook="tracepoint",
                   rarity=3.2, confidence=1.5)
    t0 = time.time()
    for i in range(n):
        ev.pid = 4000 + (i % 30000)
        await bus.publish(Topic.HOST_EVENT, ev)
    t1 = time.time()
    await asyncio.sleep(0.5)                 # let the socket flush before closing
    bus.close()
    return t0, t1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=100_000)
    ap.add_argument("--no-curve", action="store_true")
    args = ap.parse_args()
    n, curve = args.events, not args.no_curve

    with tempfile.TemporaryDirectory() as d:
        xsub, xpub = f"tcp://127.0.0.1:{free_port()}", f"tcp://127.0.0.1:{free_port()}"
        broker_cmd = [sys.executable, str(ROOT / "run_broker.py"), "--transport", "zmq",
                      "--xsub", xsub, "--xpub", xpub]
        pub_keys = sub_keys = None
        if curve:
            generate_keypair(d, "server")
            clients = os.path.join(d, "clients")
            os.makedirs(clients)
            for name in ("c1", "c3"):
                public, _ = generate_keypair(d, name)
                os.replace(public, os.path.join(clients, f"{name}.key"))
            server_pub = os.path.join(d, "server.key")
            pub_keys = (os.path.join(d, "c1.key_secret"), server_pub)
            sub_keys = (os.path.join(d, "c3.key_secret"), server_pub)
            broker_cmd += ["--curve-dir", d]
        else:
            broker_cmd += ["--insecure"]

        broker = subprocess.Popen(broker_cmd, stdout=subprocess.DEVNULL, cwd=str(ROOT))
        time.sleep(1.0)
        ready, result = mp.Event(), mp.Queue()
        sub = mp.Process(target=subscriber, args=(xsub, xpub, sub_keys, n, ready, result))
        sub.start()
        ready.wait(10)
        time.sleep(0.5)
        t0, t1 = asyncio.run(publish(xsub, xpub, pub_keys, n))
        received, _first, last = result.get(timeout=150)
        sub.join(10)
        broker.terminate()
        broker.wait(10)

    publish_rate = n / (t1 - t0)
    e2e = received / ((last or t1) - t0)
    loss = 100.0 * (n - received) / n
    print(f"  transport            protobuf over ZeroMQ, CURVE {'on' if curve else 'off'}, 3 processes")
    print(f"  events sent          {n}")
    print(f"  events received      {received}")
    print(f"  publish rate         {publish_rate:,.0f} events/s")
    print(f"  end-to-end rate      {e2e:,.0f} events/s")
    print(f"  loss                 {loss:.3f} %")
    ok = e2e >= 10_000 and received == n
    print(f"  NFR6 (>= 10,000 events/s, no loss)   {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
