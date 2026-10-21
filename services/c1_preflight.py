"""
Component 1 preflight: can this host run the agent, and with what?
IT23184312 | Amantha M A

    laax-agent check

Each check is OK, WARN (the agent runs, with a capability missing) or FAIL
(the agent cannot run in eBPF mode). The agent itself uses lsm_status() at
start-up to decide whether enforcement can be offered, so a kernel without
the BPF LSM degrades to telemetry-only instead of failing to load at all.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import List, Optional, Tuple

from services.ebpf_source import can_load_bpf, find_loader

OK, WARN, FAIL = "OK", "WARN", "FAIL"
Check = Tuple[str, str, str]

# Kernel functions the deep tier attaches to (ebpf/laax.bpf.c).
DEEP_FUNCTIONS = ("finalize_exec", "tcp_sendmsg", "inet_csk_accept", "security_file_open")
LSM_FILE = "/sys/kernel/security/lsm"


def mount_securityfs() -> None:
    """The active-LSM list lives in securityfs, which minimal systems (WSL2
    among them) do not mount by default. Mounting it changes nothing else."""
    if os.path.exists(LSM_FILE) or os.geteuid() != 0:
        return
    subprocess.run(["mount", "-t", "securityfs", "securityfs", "/sys/kernel/security"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def lsm_status() -> Tuple[Optional[bool], str]:
    """(True, list) if bpf is an active LSM, (False, list) if not, (None, why)
    if it cannot be determined."""
    mount_securityfs()
    try:
        with open(LSM_FILE) as fh:
            active = fh.read().strip()
    except OSError as exc:
        return None, f"cannot read {LSM_FILE}: {exc.strerror}"
    return "bpf" in active.split(","), active


def _kernel_symbols(names) -> List[str]:
    """Names from `names` that are NOT in /proc/kallsyms."""
    wanted = set(names)
    try:
        with open("/proc/kallsyms") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3 and parts[2] in wanted:
                    wanted.discard(parts[2])
                    if not wanted:
                        break
    except OSError:
        return list(names)
    return sorted(wanted)


def _kernel_version() -> Tuple[int, int]:
    try:
        major, minor = platform.release().split(".")[:2]
        return int(major), int("".join(c for c in minor if c.isdigit()) or 0)
    except ValueError:
        return 0, 0


def run_checks(config_path: Optional[str] = None,
               loader_path: Optional[str] = None) -> List[Check]:
    out: List[Check] = []
    if not sys.platform.startswith("linux"):
        return [("platform", FAIL, f"{sys.platform}: eBPF mode needs Linux; "
                                   "simulated mode only")]

    ver = _kernel_version()
    out.append(("kernel", OK if ver >= (5, 8) else FAIL,
                f"{platform.release()} (needs >= 5.8 for ring buffer + CAP_BPF)"))

    btf = os.path.exists("/sys/kernel/btf/vmlinux")
    out.append(("BTF", OK if btf else FAIL,
                "/sys/kernel/btf/vmlinux present" if btf
                else "missing: kernel built without CONFIG_DEBUG_INFO_BTF (CO-RE impossible)"))

    priv = can_load_bpf()
    out.append(("privileges", OK if priv else FAIL,
                "root or CAP_BPF+CAP_PERFMON+CAP_SYS_ADMIN" if priv
                else "run as root or grant CAP_BPF, CAP_PERFMON, CAP_SYS_ADMIN"))

    loader = find_loader(loader_path)
    exe = loader.exists() and os.access(loader, os.X_OK)
    out.append(("loader", OK if exe else FAIL,
                str(loader) if exe else f"{loader} missing or not executable"))

    missing = _kernel_symbols(DEEP_FUNCTIONS)
    out.append(("deep probes", OK if not missing else WARN,
                "all 4 attach points present" if not missing
                else f"missing {', '.join(missing)}: deep mode will fail to attach"))

    lsm, detail = lsm_status()
    if lsm:
        out.append(("BPF LSM", OK, f"active ({detail})"))
    else:
        why = detail if lsm is None else f"bpf not in active list ({detail}); add lsm=...,bpf to the kernel cmdline"
        out.append(("BPF LSM", WARN, f"enforcement unavailable: {why}"))

    try:
        import zmq  # noqa: F401
        from laax import laax_pb2  # noqa: F401
        out.append(("zmq transport", OK, f"pyzmq {zmq.pyzmq_version()}, CURVE "
                                         f"{'yes' if zmq.has('curve') else 'NO'}"))
    except ImportError as exc:
        out.append(("zmq transport", WARN, f"unavailable ({exc.name} missing); "
                                           "install python3-zmq python3-protobuf"))

    if config_path:
        out.append(_file_check("config", config_path, secret=False))
    return out


def _fs_type(path: str) -> str:
    """Filesystem type of the mount holding path (longest matching mount point)."""
    path = os.path.realpath(path)
    best, fstype = "", ""
    try:
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                if (path == mnt or path.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
                    best, fstype = mnt, parts[2]
    except OSError:
        pass
    return fstype


# Windows drives under WSL: every file reports 0777 and chmod does nothing.
NO_UNIX_PERMS = {"9p", "drvfs", "v9fs", "vfat", "ntfs", "ntfs3", "fuseblk"}


def _file_check(label: str, path: str, secret: bool) -> Check:
    try:
        mode = os.stat(path).st_mode & 0o777
    except OSError as exc:
        return label, FAIL, f"{path}: {exc.strerror}"
    fstype = _fs_type(path)
    if fstype in NO_UNIX_PERMS:
        return label, WARN, (f"{path} is on {fstype}, which does not enforce Unix "
                             f"permissions; install it under /etc/laax for production")
    if mode & 0o002:
        return label, FAIL, f"{path} is world-writable (mode {mode:o})"
    if secret and mode & 0o077:
        return label, FAIL, f"{path} readable by others (mode {mode:o})"
    return label, OK, f"{path} (mode {mode:o})"


def report(checks: List[Check]) -> int:
    for name, status, detail in checks:
        print(f"  {status:<5} {name:<15} {detail}")
    if any(s == FAIL for _, s, _ in checks):
        print("\n  eBPF mode cannot run on this host.")
        return 1
    if any(s == WARN for _, s, _ in checks):
        print("\n  eBPF mode can run, with the limitations marked WARN.")
    else:
        print("\n  All checks passed.")
    return 0
