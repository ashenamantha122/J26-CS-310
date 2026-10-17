#!/usr/bin/env bash
# check_ebpf_kernel.sh — run this FIRST, before writing a single line of
# eBPF/LSM code. If this script says NO anywhere, stop and fix the
# environment (or pivot the attach point) before continuing; everything
# downstream depends on this passing.

set -uo pipefail
PASS="✅"
FAIL="❌"
WARN="⚠️ "

echo "=== Component 4 eBPF/LSM kernel compatibility check ==="
echo

# 1. Kernel version >= 5.7 (minimum for BPF LSM)
KVER=$(uname -r | grep -oE '^[0-9]+\.[0-9]+')
KMAJOR=$(echo "$KVER" | cut -d. -f1)
KMINOR=$(echo "$KVER" | cut -d. -f2)
echo "Kernel version: $(uname -r)"
if [ "$KMAJOR" -gt 5 ] || { [ "$KMAJOR" -eq 5 ] && [ "$KMINOR" -ge 7 ]; }; then
    echo "$PASS kernel >= 5.7"
else
    echo "$FAIL kernel < 5.7 — BPF LSM hooks are not available. You will need a newer"
    echo "   kernel (a current Kali/Debian/Ubuntu point release, or a backport),"
    echo "   or pivot to cgroup/syscall-BPF attach points instead of LSM hooks."
fi
echo

# 2. CONFIG_BPF_LSM=y in the running kernel config
CONFIG_SRC=""
if [ -f "/boot/config-$(uname -r)" ]; then
    CONFIG_SRC="/boot/config-$(uname -r)"
elif [ -f /proc/config.gz ]; then
    CONFIG_SRC="/proc/config.gz"
fi

if [ -n "$CONFIG_SRC" ]; then
    if [[ "$CONFIG_SRC" == *.gz ]]; then
        VAL=$(zcat "$CONFIG_SRC" | grep -E '^CONFIG_BPF_LSM=' || true)
    else
        VAL=$(grep -E '^CONFIG_BPF_LSM=' "$CONFIG_SRC" || true)
    fi
    if [ "$VAL" = "CONFIG_BPF_LSM=y" ]; then
        echo "$PASS CONFIG_BPF_LSM=y ($CONFIG_SRC)"
    else
        echo "$FAIL CONFIG_BPF_LSM is not set to y in $CONFIG_SRC"
        echo "   This is the most likely blocker. Options: enable it and rebuild/reboot"
        echo "   (distro-dependent), use a distro kernel that ships it enabled (recent"
        echo "   Ubuntu/Fedora server kernels usually do), or pivot the enforcer to"
        echo "   cgroup/BPF or seccomp-BPF instead of LSM hooks for the demo."
    fi
else
    echo "$WARN could not locate kernel config (/boot/config-* or /proc/config.gz) —"
    echo "   checking CONFIG_BPF_LSM via /sys instead (less definitive)"
fi
echo

# 3. BPF LSM actually active for this boot (not just compiled in — must be in the LSM list)
if [ -f /sys/kernel/security/lsm ]; then
    LSMS=$(cat /sys/kernel/security/lsm)
    echo "Active LSMs this boot: $LSMS"
    if echo "$LSMS" | grep -qw bpf; then
        echo "$PASS 'bpf' is in the active LSM list"
    else
        echo "$FAIL 'bpf' is NOT in the active LSM list, even if compiled in."
        echo "   Fix: add 'lsm=...,bpf' to the kernel cmdline (GRUB_CMDLINE_LINUX in"
        echo "   /etc/default/grub), run update-grub, and reboot."
    fi
else
    echo "$FAIL /sys/kernel/security/lsm not present — LSM framework may be disabled entirely."
fi
echo

# 4. BTF availability (needed for CO-RE — portable eBPF without per-kernel compilation)
if [ -f /sys/kernel/btf/vmlinux ]; then
    echo "$PASS BTF available at /sys/kernel/btf/vmlinux"
else
    echo "$FAIL No BTF at /sys/kernel/btf/vmlinux — CO-RE eBPF won't work out of the box."
    echo "   You can still build with a full kernel-headers-based compile instead of CO-RE,"
    echo "   but that's slower to iterate on. Installing a BTF-enabled kernel is preferred."
fi
echo

# 5. Required tools present
for tool in bpftool clang llc; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "$PASS $tool found: $(command -v "$tool")"
    else
        echo "$FAIL $tool not found — install with your package manager"
        echo "   (Kali/Debian: apt install bpftool clang llvm)"
    fi
done
echo

echo "=== Summary ==="
echo "If every line above is $PASS, proceed straight to writing the minimal"
echo "LSM hook program (hook_skeleton.bpf.c). If anything is $FAIL,"
echo "resolve it today — this is the single biggest schedule risk before Oct 20."
