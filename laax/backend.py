"""
Tracing backends for Component 1.

The mode controller decides WHEN to change tracing depth. A backend actually
does it and reports how long it took. Two implementations:

    SimulatedBackend  -- a cost model. Runs anywhere, including Windows.
    EbpfBackend       -- drives the real laax-loader over a pipe. Numbers are
                         measurements. See services/ebpf_source.py.

Every method returns latency in milliseconds, because that is the quantity the
proposal budgets and therefore the quantity that has to be reported.
"""

from __future__ import annotations

from typing import Protocol


class TracingBackend(Protocol):
    """What Component 1 needs from a kernel, and nothing more."""

    async def attach_deep(self) -> float: ...
    async def detach_deep(self) -> float: ...
    async def gate_add(self, pid: int) -> float: ...
    async def gate_del(self, pid: int) -> float: ...
    async def deny_add(self, pid: int, syscall: str) -> float: ...
    async def deny_del(self, pid: int) -> float: ...
    async def dropped(self) -> int: ...


class SimulatedBackend:
    """
    Cost model, not measurement. Any number produced through this backend must
    be labelled as modelled when it is reported.

    The defaults below are the MEDIANS measured by `run_live.py --calibrate` on
    WSL2 kernel 6.18.33.2, n=30, with the four-probe deep tier (finalize_exec,
    tcp_sendmsg, inet_csk_accept, security_file_open) of 21 Oct 2026:

        bpf_link attach (deep)      26.46 ms
        bpf_link destroy (deep)    371.08 ms
        map update (gate a pid)      0.008 ms

    Note the asymmetry, which the first version of this model had backwards.
    Detaching a kprobe must unregister it and wait out an RCU grace period;
    attaching does not. The kernel serialises those waits (~95 ms per probe,
    measured individually), so detach cost grows linearly with the number of
    deep probes. Leaving deep mode is roughly 14x more expensive than entering
    it, so the minimum dwell time is not merely an anti-flap heuristic -- it
    has a direct cost justification.
    """

    measured = False
    enforcement_available = True

    def __init__(self, attach_ms: float = 26.46, detach_ms: float = 371.08,
                 map_ms: float = 0.008) -> None:
        self.attach_ms, self.detach_ms, self.map_ms = attach_ms, detach_ms, map_ms
        self.deep_attached = False
        self._dropped = 0

    async def attach_deep(self) -> float:
        self.deep_attached = True
        return self.attach_ms

    async def detach_deep(self) -> float:
        self.deep_attached = False
        return self.detach_ms

    async def gate_add(self, pid: int) -> float:
        return self.map_ms

    async def gate_del(self, pid: int) -> float:
        return self.map_ms

    async def deny_add(self, pid: int, syscall: str) -> float:
        # Same contract as EbpfBackend: the LSM hook only covers execve.
        if syscall != "execve":
            raise ValueError(f"LSM hook denies execve only, not {syscall!r}")
        return self.map_ms

    async def deny_del(self, pid: int) -> float:
        return self.map_ms

    async def dropped(self) -> int:
        return self._dropped
