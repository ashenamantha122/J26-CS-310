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

LOADER = Path(__file__).resolve().parent.parent / "ebpf" / "laax-loader"


class LoaderUnavailable(RuntimeError):
    pass


class EbpfLoader:
    """Owns the laax-loader subprocess and demultiplexes its output."""

    def __init__(self, path: Path = LOADER, lsm: bool = False,
                 sudo: Optional[bool] = None) -> None:
        self.path, self.lsm = Path(path), lsm
        # Loading BPF programs needs CAP_BPF/CAP_SYS_ADMIN. Use sudo unless we
        # are already root.
        self.sudo = (os.geteuid() != 0) if sudo is None else sudo
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.events: asyncio.Queue = asyncio.Queue()
        self.drops = 0
        self.ready = asyncio.Event()
        self._pending: Optional[asyncio.Future] = None
        self._cmd_lock = asyncio.Lock()
        self._reader: Optional[asyncio.Task] = None

    # -- lifecycle ---------------------------------------------------------
    async def start(self, timeout: float = 10.0) -> None:
        if not self.path.exists():
            raise LoaderUnavailable(
                f"{self.path} not built. Run: cd ebpf && make")
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
        try:
            await asyncio.wait_for(self.ready.wait(), timeout)
        except asyncio.TimeoutError:
            err = b""
            if self.proc.stderr:
                try:
                    err = await asyncio.wait_for(self.proc.stderr.read(4096), 1.0)
                except asyncio.TimeoutError:
                    pass
            raise LoaderUnavailable(
                f"loader did not become ready: {err.decode(errors='replace').strip()}")

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                await self._send("QUIT")
                await asyncio.wait_for(self.proc.wait(), 3.0)
            except (asyncio.TimeoutError, ProcessLookupError, BrokenPipeError):
                self.proc.kill()
        if self._reader:
            self._reader.cancel()

    # -- output demux ------------------------------------------------------
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
        # the hook it is attached to. Widening this means another hook.
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
        ev = HostEvent(
            host_id=self.host_id,
            pid=d["pid"],
            ppid=d["ppid"],
            comm=d["comm"],
            kind=d["kind"],
            depth=d["depth"],
            # laax.bpf.c: every light-tier program is a tracepoint and every
            # deep-tier program is a kprobe, so the tier identifies the hook.
            hook="kprobe" if d["depth"] == "deep" else "tracepoint",
            saddr=d["saddr"] or None,
            daddr=d["daddr"] or None,
            dport=d["dport"] or None,
            proto="tcp" if d["kind"] in ("connect", "accept") else None,
            argv=[d["argv"]] if d["argv"] else None,
            syscall={"exec": "execve", "connect": "connect",
                     "accept": "accept4"}.get(d["kind"]),
            ret=d["ret"] or None,
            source="ebpf",
        )
        # The kernel gives us the parent comm directly on exec, which saves the
        # agent a pid->comm lookup that races against process exit.
        if d.get("pcomm"):
            ev.comm_parent = d["pcomm"]  # type: ignore[attr-defined]
        return ev
