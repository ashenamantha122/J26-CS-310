"""
Live-kernel checks for the fork/exit fixes and the deep tier. Linux, root,
needs bpf in the active LSM list.

    python3 scenario/test_inheritance.py

Before 21 Oct 2026 both of these failed on the live kernel:
  - a gated process's forked child exec'd without a deep event (the live
    agent run captured 0 deep payloads), and
  - a denied process could fork a child that exec'd freely (rc=0).

Cases
  1. deep:  forked child of a gated process -> deep exec event with full argv
  2. deep:  accept and file-open events for a gated process
  3. lsm:   denied process -> its forked child's exec is refused
  4. lsm:   after release -> the same fork+exec is allowed
  5. exit:  a denied process exits -> its denied_pids entry is gone

Same root-namespace pid trick as test_lsm.py: the target announces its kernel
pid by exec'ing a uniquely named marker binary, whose exec event carries it as
ppid.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.ebpf_source import EbpfBackend, EbpfLoader, LoaderUnavailable

TARGET = r"""
import os, socket, subprocess, sys
subprocess.run([{marker!r}])
print("announced", flush=True)
for line in sys.stdin:
    cmd = line.strip()
    if cmd == "forkexec":
        try:
            r = subprocess.run(["/bin/echo", "laax", "child-arg", "two words"],
                               capture_output=True)
            print("rc=%d" % r.returncode, flush=True)
        except PermissionError:
            print("EPERM", flush=True)
    elif cmd == "netfile":
        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
        c = socket.create_connection(srv.getsockname())
        conn, _ = srv.accept()
        c.sendall(b"x")
        open("/etc/hostname").read()
        conn.close(); c.close(); srv.close()
        print("done", flush=True)
    elif cmd == "exit":
        break
"""


class Target:
    def __init__(self, loader: EbpfLoader, marker: str) -> None:
        self.loader, self.marker = loader, marker

    async def start(self) -> int:
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", TARGET.format(marker=self.marker),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        comm = os.path.basename(self.marker)[:15]
        while True:
            ev = await asyncio.wait_for(self.loader.events.get(), 5.0)
            if ev.get("comm") == comm and ev.get("ppid"):
                self.pid = int(ev["ppid"])
                break
        await self.proc.stdout.readline()
        return self.pid

    async def do(self, cmd: str) -> str:
        self.proc.stdin.write((cmd + "\n").encode())
        await self.proc.stdin.drain()
        if cmd == "exit":
            await self.proc.wait()
            return ""
        return (await self.proc.stdout.readline()).decode().strip()


async def collect(loader: EbpfLoader, seconds: float = 0.5) -> list:
    out, loop = [], asyncio.get_running_loop()
    end = loop.time() + seconds
    while (left := end - loop.time()) > 0:
        try:
            out.append(await asyncio.wait_for(loader.events.get(), left))
        except asyncio.TimeoutError:
            break
    return out


def drain(loader: EbpfLoader) -> None:
    while not loader.events.empty():
        loader.events.get_nowait()


def denied_pids() -> set:
    """Keys of the denied_pids map, read with bpftool (independent of LAAX)."""
    try:
        out = subprocess.run(["bpftool", "map", "dump", "name", "denied_pids", "-j"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {-1}
    import json
    keys = set()
    for entry in json.loads(out or "[]"):
        k = entry.get("key")
        if isinstance(k, list):               # raw bytes, little-endian u32
            keys.add(int.from_bytes(bytes(int(b, 16) for b in k), "little"))
        else:
            keys.add(int(k))
    return keys


def report(label: str, ok: bool, detail: str) -> bool:
    print(f"  {label:<46} {'PASS' if ok else 'FAIL':<5} {detail}")
    return ok


async def main() -> int:
    marker = f"/tmp/lxi{uuid.uuid4().hex[:6]}"
    shutil.copy("/bin/true", marker)
    loader = EbpfLoader(lsm=True)
    try:
        await loader.start()
    except LoaderUnavailable as exc:
        print(f"cannot start loader with --lsm: {exc}")
        return 1
    be = EbpfBackend(loader)
    results = []
    print("\n  Fork inheritance, exit cleanup and the deep tier on a live kernel\n")

    # -- deep tier ---------------------------------------------------------
    t = Target(loader, marker)
    pid = await t.start()
    await be.gate_add(pid)
    await be.attach_deep()
    drain(loader)
    await t.do("forkexec")
    evs = await collect(loader)
    child = [e for e in evs if e["kind"] == "exec" and e["depth"] == "deep"
             and e["comm"] == "echo"]
    fork = [e for e in evs if e["kind"] == "fork" and e["ppid"] == pid]
    argv = child[0]["argv"].split("\x1f") if child else []
    results.append(report("1 deep: forked child's exec captured",
                          bool(child) and bool(fork),
                          f"argv={argv}" if child else "no deep exec event"))
    results.append(report("  ...with argv entries intact",
                          argv == ["/bin/echo", "laax", "child-arg", "two words"], ""))

    drain(loader)
    await t.do("netfile")
    evs = [e for e in await collect(loader) if e["pid"] == pid and e["depth"] == "deep"]
    kinds = {e["kind"] for e in evs}
    files = [e["argv"] for e in evs if e["kind"] == "file"]
    results.append(report("2 deep: accept + send + file-open events",
                          {"accept", "connect", "file"} <= kinds,
                          f"kinds={sorted(kinds)} hostname_seen={'hostname' in files}"))
    await t.do("exit")
    await be.gate_del(pid)
    await be.detach_deep()

    # -- LSM -----------------------------------------------------------------
    t = Target(loader, marker)
    pid = await t.start()
    await be.deny_add(pid, "execve")
    out = await t.do("forkexec")
    results.append(report("3 lsm: denied process's forked child refused",
                          out == "EPERM", f"child exec -> {out}"))
    await be.deny_del(pid)
    out = await t.do("forkexec")
    results.append(report("4 lsm: after release the fork+exec is allowed",
                          out == "rc=0", f"child exec -> {out}"))

    await be.deny_add(pid, "execve")
    before = pid in denied_pids()
    await t.do("exit")
    await asyncio.sleep(0.2)
    after = pid in denied_pids()
    results.append(report("5 exit: denied_pids entry removed on exit",
                          before and not after,
                          f"in map before exit={before}, after={after}"))

    await loader.stop()
    os.unlink(marker)
    ok = all(results)
    print(f"\n  {'all cases behaved as specified' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
