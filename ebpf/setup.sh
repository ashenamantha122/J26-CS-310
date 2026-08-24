#!/usr/bin/env bash
# One-shot environment setup for the LAAX Component 1 kernel build.
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/e/Accepted\ RP/Prototype/ebpf/setup.sh
set -euo pipefail

echo "== kernel =="
uname -r

echo
echo "== installing toolchain =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    clang llvm libbpf-dev libelf-dev zlib1g-dev make gcc python3 python3-venv

# Does a WORKING bpftool exist? `command -v` is not enough: linux-tools-generic
# installs a wrapper at /usr/sbin/bpftool that dispatches to a per-kernel-version
# binary, and on WSL2 that binary does not exist, so the wrapper is on PATH and
# still fails. Test that it actually runs.
have_bpftool() { bpftool version >/dev/null 2>&1; }

if ! have_bpftool; then
    apt-get install -y -qq bpftool 2>/dev/null || true
fi
if ! have_bpftool; then
    apt-get install -y -qq "linux-tools-$(uname -r)" 2>/dev/null || true
fi

# WSL2 kernels are not in the apt archive, so linux-tools-$(uname -r) will never
# resolve. Build bpftool from source against the running kernel's BTF.
if ! have_bpftool; then
    echo "-- no working bpftool for kernel $(uname -r); building from source"
    # libssl-dev is not optional: bpftool's sign.c includes <openssl/opensslv.h>.
    apt-get install -y -qq git bison flex libcap-dev pkg-config libssl-dev
    rm -rf /tmp/bpftool
    git clone --quiet --depth 1 --recurse-submodules \
        https://github.com/libbpf/bpftool.git /tmp/bpftool
    make -s -C /tmp/bpftool/src -j"$(nproc)"
    install -m 0755 /tmp/bpftool/src/bpftool /usr/local/bin/bpftool
    hash -r
fi

have_bpftool || { echo "ERROR: still no working bpftool"; exit 1; }

echo
echo "== versions =="
clang --version | head -1
bpftool version | head -1

echo
echo "== capability check =="
cd "$(dirname "$0")"
make check || true

echo
echo "== build =="
make

echo
echo "Built: $(pwd)/laax-loader"
echo "Next:  sudo python3 ../run_live.py --calibrate"
