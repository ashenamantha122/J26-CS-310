"""
Real-kernel Component 1: drives ebpf/laax-loader over a pipe.

The loader owns the BPF objects; this module owns the conversation with it.
One stdout stream carries two kinds of line, so a single reader task
demultiplexes:

    {...}            an event      -> the event queue, consumed by EbpfKernelSource
    #LAT <op> <ms>   a measurement -> resolves the pending control future
    #DROP <n>        ring-buffer reservation failures
    #READY           light mode attached, deep loaded but unattached

Swapping this in for the scripted source is the only change Component 1 needs
to go from a cost model to measurements:

    loader = EbpfLoader(); await loader.start()
    agent  = C1Agent(bus, EbpfKernelSource(loader), backend=EbpfBackend(loader))
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from laax.contracts import HostEvent

REPO_LOADER = Path(__file__).resolve().parent.parent / "ebpf" / "laax-loader"
PACKAGED_LOADER = Path("/usr/lib/laax/laax-loader")


def find_loader(explicit: Optional[str] = None) -> Path:
    """$LAAX_LOADER, then an explicit path, then the source tree, then the
    packaged location. The first that exists wins; the packaged path is
    returned as the default so error messages name where it should be."""
    for candidate in (os.environ.get("LAAX_LOADER"), explicit):
        if candidate:
            return Path(candidate)
    for candidate in (REPO_LOADER, PACKAGED_LOADER):
        if candidate.exists():
            return candidate
    return PACKAGED_LOADER


LOADER = find_loader()

# Linux capability bit numbers (linux/capability.h).
CAP_SYS_ADMIN, CAP_PERFMON, CAP_BPF = 21, 38, 39


def effective_caps() -> int:
    """CapEff of this process, 0 where /proc is unavailable."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("CapEff:"):
                    return int(line.split()[1], 16)
    except OSError:
        pass
    return 0


def can_load_bpf() -> bool:
    """Root, or the capabilities the loader needs: CAP_BPF and CAP_PERFMON for
    the programs and kprobes, CAP_SYS_ADMIN for the LSM program. A systemd
    unit grants them as ambient capabilities, which the loader inherits."""
    if os.geteuid() == 0:
        return True
    caps = effective_caps()
    return all(caps & (1 << c) for c in (CAP_BPF, CAP_PERFMON, CAP_SYS_ADMIN))


class LoaderUnavailable(RuntimeError):
    pass


class EbpfLoader:
    """Owns the laax-loader subprocess and demultiplexes its output."""

    def __init__(self, path: Optional[Path] = None, lsm: bool = False,
                 sudo: Optional[bool] = None) -> None:
        self.path, self.lsm = Path(path) if path else find_loader(), lsm
        # Use sudo only when this process can neither load BPF itself nor pass
        # the capabilities on to the loader.
        self.sudo = (not can_load_bpf()) if sudo is None else sudo
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.events: asyncio.Queue = asyncio.Queue()
        self.drops = 0
        self.ready = asyncio.Event()
        self._pending: Optional[asyncio.Future] = None
        self._cmd_lock = asyncio.Lock()
        self._reader: Optional[asyncio.Task] = None
        self._err_reader: Optional[asyncio.Task] = None
        self.stderr_lines: list[str] = []

    # -- lifecycle ---------------------------------------------------------
    async def start(self, timeout: float = 10.0) -> None:
        if not self.path.exists():
            raise LoaderUnavailable(
                f"{self.path} not built. Run: cd ebpf && make (or set LAAX_LOADER)")
        argv = [str(self.path)] + (["--lsm"] if self.lsm else [])
        if self.sudo:
            if not shutil.which("sudo"):
                raise LoaderUnavailable("not root and sudo unavailable")
            argv = ["sudo", "-n"] + argv

        self.proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader = asyncio.create_task(self._read_loop())
        self._err_reader = asyncio.create_task(self._stderr_loop())
        try:
            await asyncio.wait_for(self.ready.wait(), timeout)
        except asyncio.TimeoutError:
            await asyncio.sleep(0.2)            # let the stderr reader catch up
            raise LoaderUnavailable(
                f"loader did not become ready: {' | '.join(self.stderr_lines[-5:])}")

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                await self._send("QUIT")
                await asyncio.wait_for(self.proc.wait(), 3.0)
            except (asyncio.TimeoutError, ProcessLookupError, BrokenPipeError):
                self.proc.kill()
        for task in (self._reader, self._err_reader):
            if task:
                task.cancel()

    # -- output demux ------------------------------------------------------
    async def _stderr_loop(self) -> None:
        """Keep the loader's stderr drained. An unread pipe fills at 64 KiB and
        the loader then blocks inside fprintf -- and stops collecting."""
        assert self.proc and self.proc.stderr
        while True:
            raw = await self.proc.stderr.readline()
            if not raw:
                return
            line = raw.decode(errors="replace").strip()
            if line:
                self.stderr_lines = (self.stderr_lines + [line])[-50:]

    async def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            raw = await self.proc.stdout.readline()
            if not raw:
                return
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    await self.events.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
            elif line.startswith("#LAT "):
                _, _op, ms = line.split(None, 2)
                if self._pending and not self._pending.done():
                    self._pending.set_result(float(ms))
            elif line.startswith("#DROP "):
                self.drops = int(line.split()[1])
            elif line.startswith("#READY"):
                self.ready.set()
            elif line.startswith("#ERR"):
                if self._pending and not self._pending.done():
                    self._pending.set_exception(RuntimeError(line))

    # -- control -----------------------------------------------------------
    async def _send(self, cmd: str) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write((cmd + "\n").encode())
        await self.proc.stdin.drain()

    async def command(self, cmd: str, timeout: float = 5.0) -> float:
        """Send a control command and return the loader's measured latency."""
        async with self._cmd_lock:
            self._pending = asyncio.get_running_loop().create_future()
            await self._send(cmd)
            try:
                return await asyncio.wait_for(self._pending, timeout)
            finally:
                self._pending = None

    async def refresh_drops(self) -> int:
        assert self.proc and self.proc.stdin
        await self._send("STATS")
        await asyncio.sleep(0.05)
        return self.drops


class EbpfBackend:
    """TracingBackend over a live kernel. Every number here is measured."""

    measured = True

    def __init__(self, loader: EbpfLoader) -> None:
        self.loader = loader

    @property
    def enforcement_available(self) -> bool:
        # Without --lsm the denial program is not loaded: a DENY_ADD would
        # write a map nobody reads, and report containment that does nothing.
        return self.loader.lsm

    async def attach_deep(self) -> float:
        return await self.loader.command("ATTACH_DEEP")

    async def detach_deep(self) -> float:
        return await self.loader.command("DETACH_DEEP")

    async def gate_add(self, pid: int) -> float:
        return await self.loader.command(f"GATE_ADD {pid}")

    async def gate_del(self, pid: int) -> float:
        return await self.loader.command(f"GATE_DEL {pid}")

    async def deny_add(self, pid: int, syscall: str) -> float:
        # The LSM program keys on pid alone; the syscall is fixed at execve by
        # the hook it is attached to. Widening this means another hook, so
        # anything else is refused rather than silently armed as execve.
        if syscall != "execve":
            raise ValueError(f"LSM hook denies execve only, not {syscall!r}")
        return await self.loader.command(f"DENY_ADD {pid}")

    async def deny_del(self, pid: int) -> float:
        return await self.loader.command(f"DENY_DEL {pid}")

    async def dropped(self) -> int:
        return await self.loader.refresh_drops()


class EbpfKernelSource:
    """Async iterator of HostEvent, fed by the loader's ring buffer."""

    def __init__(self, loader: EbpfLoader, host_id: str = "host-0") -> None:
        self.loader, self.host_id = loader, host_id

    def __aiter__(self):
        return self

    async def __anext__(self) -> HostEvent:
        d = await self.loader.events.get()
        kind = d["kind"]
        # fork and exit are process-lifecycle bookkeeping from light-tier
        # tracepoints; a fork event's depth says whether the child was born
        # deep, not which hook fired.
        deep_hook = d["depth"] == "deep" and kind not in ("fork", "exit")
        argv = d.get("argv") or ""
        ev = HostEvent(
            host_id=self.host_id,
            pid=d["pid"],
            ppid=d["ppid"],
            comm=d["comm"],
            exe=d.get("filename") or None,
            kind=kind,
            depth=d["depth"],
            # laax.bpf.c: every light-tier program is a tracepoint and every
            # deep-tier program is a kprobe or kretprobe, so the tier
            # identifies the hook.
            hook="kprobe" if deep_hook else "tracepoint",
            saddr=d["saddr"] or None,
            daddr=d["daddr"] or None,
            sport=d.get("sport") or None,
            dport=d["dport"] or None,
            # From the socket's type in-kernel; empty when it could not be
            # determined, which is recorded as unknown rather than guessed.
            proto=(d.get("proto") or None) if kind in ("connect", "accept") else None,
            # The loader separates argv entries with 0x1f (the kernel's NULs).
            argv=argv.split("\x1f") if argv else None,
            syscall={"exec": "execve", "connect": "connect",
                     "accept": "accept4", "file": "open"}.get(kind),
            ret=d["ret"] or None,
            source="ebpf",
        )
        # The kernel gives us the parent comm directly on exec, which saves the
        # agent a pid->comm lookup that races against process exit.
        if d.get("pcomm"):
            ev.comm_parent = d["pcomm"]  # type: ignore[attr-defined]
        if d.get("gpcomm"):
            ev.comm_grandparent = d["gpcomm"]  # type: ignore[attr-defined]
        if kind == "fork":
            ev.inherit_flags = d.get("flags", 0)  # type: ignore[attr-defined]
        return ev
