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
    WSL2 kernel 6.18.33.2, n=30:

        bpf_link attach (deep)      13.73 ms
        bpf_link destroy (deep)    174.80 ms
        map update (gate a pid)      0.011 ms

    Note the asymmetry, which the first version of this model had backwards.
    Detaching a kprobe must unregister it and wait out an RCU grace period;
    attaching does not. Leaving deep mode is roughly 13x more expensive than
    entering it, so the minimum dwell time is not merely an anti-flap heuristic
    -- it has a direct cost justification.
    """

    measured = False

    def __init__(self, attach_ms: float = 13.73, detach_ms: float = 174.80,
                 map_ms: float = 0.011) -> None:
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
        return self.map_ms

    async def deny_del(self, pid: int) -> float:
        return self.map_ms

    async def dropped(self) -> int:
        return self._dropped
