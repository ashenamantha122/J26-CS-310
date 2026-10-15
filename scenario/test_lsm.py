"""
Does the eBPF LSM hook actually deny an execve? Linux, root, needs bpf in the
active LSM list.

    python3 scenario/test_lsm.py

Two things this test has to get right, both of which caught me out.

1. bprm_check_security runs in the context of the process performing the exec.
   After a fork the CHILD execs under its own new pid, so denying the parent's
   pid proves nothing. The target therefore calls os.execv on ITSELF, keeping
   the pid constant -- which is what denied_pids keys on.

2. bpf_get_current_pid_tgid() returns the pid in the ROOT pid namespace. This
   WSL2 distro runs in a child namespace, where the same process was pid 389
   locally and 4984 to the kernel. Arming a denial with os.getpid() silently
   never matches. So the target announces its root-namespace pid the only way
   it reliably can: by spawning a uniquely-named marker binary, whose exec
   event carries the target's root-ns pid as ppid.

The second point is a property of the architecture, not of this test. C1's
telemetry is in root-namespace pids, so any pid C4 sends back must have come
from C1 in the first place. See docs/INTEGRATION.md.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.ebpf_source import EbpfBackend, EbpfLoader, LoaderUnavailable

TARGET = r"""
import os, subprocess, sys
subprocess.run([{marker!r}], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("announced", flush=True)
sys.stdin.readline()                       # wait until the verdict is armed
try:
    os.execv("/bin/true", ["/bin/true"])   # on success this image is gone
except PermissionError:
    print("EPERM", flush=True)
except Exception as exc:
    print("ERR " + type(exc).__name__, flush=True)
"""


async def root_pid_from_marker(loader: EbpfLoader, comm: str,
                               timeout: float = 5.0) -> int:
    """Find the marker's exec event; its ppid is the target's root-ns pid."""
    async def scan() -> int:
        while True:
            ev = await loader.events.get()
            if ev.get("comm") == comm and ev.get("ppid"):
                return int(ev["ppid"])
    return await asyncio.wait_for(scan(), timeout)


async def attempt(label: str, loader: EbpfLoader, backend: EbpfBackend,
                  marker: str, deny: bool) -> bool:
    comm = os.path.basename(marker)[:15]      # TASK_COMM_LEN - 1
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", TARGET.format(marker=marker),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        root_pid = await root_pid_from_marker(loader, comm)
    except asyncio.TimeoutError:
        print(f"  {label:<34} FAILED: never saw marker exec event")
        proc.kill()
        return False
    await proc.stdout.readline()              # "announced"

    latency = None
    if deny:
        latency = await backend.deny_add(root_pid, "execve")

    proc.stdin.write(b"\n")
    await proc.stdin.drain()
    out = (await proc.stdout.read()).decode().strip()
    await proc.wait()

    denied = out == "EPERM"
    ok = denied == deny
    detail = f"map write {latency:.4f}ms" if latency is not None else "no policy"
    print(f"  {label:<34} root pid {root_pid:<7} "
          f"{'DENIED' if denied else 'allowed':<8} "
          f"{'PASS' if ok else 'FAIL':<5} ({detail})")
    if deny:
        await backend.deny_del(root_pid)
    return ok


async def main() -> int:
    marker = f"/tmp/lxm{uuid.uuid4().hex[:6]}"
    shutil.copy("/bin/true", marker)
    os.chmod(marker, os.stat(marker).st_mode | stat.S_IEXEC)

    loader = EbpfLoader(lsm=True)
    try:
        await loader.start()
    except LoaderUnavailable as exc:
        print(f"cannot start loader with --lsm: {exc}")
        print("check: cat /sys/kernel/security/lsm   (needs 'bpf')")
        return 1

    backend = EbpfBackend(loader)
    print("\n  eBPF LSM enforcement -- lsm/bprm_check_security")
    print(f"  marker binary: {marker}\n")
    results = []
    for label, deny in (("no policy -> exec allowed", False),
                        ("denial armed -> exec refused", True),
                        ("denial lifted -> exec allowed", False)):
        results.append(await attempt(label, loader, backend, marker, deny))

    await loader.stop()
    os.unlink(marker)

    ok = all(results)
    print(f"\n  {'all three cases behaved as specified' if ok else 'FAILURES ABOVE'}")
    if ok:
        print("  The shell that spawned the target was never signalled and never")
        print("  terminated: the denial refused one execve and nothing else.")
        print("  That is the surgical-containment claim, demonstrated.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
