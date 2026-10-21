#!/usr/bin/env bash
# ==============================================================================
# LAAX Component 1: DEVELOPMENT install (editable pip install from this tree)
# Student: Amantha M.A (IT23184312) | Research Project: J26-CS-310
#
# For an installed, supported agent use the Debian package instead:
#   bash packaging/build-deb.sh && sudo apt install ./dist/laax-agent_*.deb
# This script is for working on the code: the agent runs from the source tree.
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo "  Installing LAAX Adaptive eBPF Endpoint Agent (Component 1)"
echo "======================================================================"

# 1. Root Check
if [ "$EUID" -ne 0 ]; then
    echo "[!] Notice: Not running as root. Installing in user mode."
    SUDO="sudo"
else
    SUDO=""
fi

# 2. Check Python
if ! command -v python3 &>/dev/null; then
    echo "[-] Error: python3 is not installed. Run: sudo apt install python3 python3-pip"
    exit 1
fi

echo "[*] Python version: $(python3 --version)"

# 3. Build eBPF loader if on Linux
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/ebpf" ]; then
    echo "[*] Building eBPF kernel objects in $SCRIPT_DIR/ebpf..."
    cd "$SCRIPT_DIR/ebpf"
    if command -v clang &>/dev/null && command -v bpftool &>/dev/null; then
        make all
        echo "[+] eBPF laax-loader built successfully."
    else
        echo "[!] clang or bpftool missing. Skipping eBPF rebuild (using prebuilt binary if present)."
    fi
    cd "$SCRIPT_DIR"
fi

# 4. Install Python Package
echo "[*] Installing laax-agent Python package and CLI binaries..."
python3 -m pip install -e "$SCRIPT_DIR"

# 5. Create System Directories
echo "[*] Setting up configuration & log directories..."
$SUDO mkdir -p /etc/laax /var/log/laax
# The audit log and telemetry are forensic records: root-only.
$SUDO chmod 0750 /var/log/laax
if [ -f "$SCRIPT_DIR/deploy/laax-agent.conf" ]; then
    $SUDO cp -n "$SCRIPT_DIR/deploy/laax-agent.conf" /etc/laax/agent.conf || true
fi

# 6. Install Systemd Service if systemctl is active
if command -v systemctl &>/dev/null && [ -d /etc/systemd/system ]; then
    AGENT_BIN="$(command -v laax-agent || true)"
    if [ -z "$AGENT_BIN" ]; then
        echo "[-] laax-agent entry point not on PATH after pip install; not installing the unit."
        exit 1
    fi
    echo "[*] Installing systemd service unit (ExecStart=$AGENT_BIN)..."
    sed "s|@LAAX_AGENT@|$AGENT_BIN|" "$SCRIPT_DIR/deploy/laax-agent.service" \
        | $SUDO tee /etc/systemd/system/laax-agent.service >/dev/null
    $SUDO systemctl daemon-reload || true
    echo "[+] Installed /etc/systemd/system/laax-agent.service"
    echo "    To enable on boot: sudo systemctl enable laax-agent"
    echo "    To start daemon:   sudo systemctl start laax-agent"
fi

echo ""
echo "======================================================================"
echo "  Installation Complete!"
echo "  You can now run:"
echo "    laax-agent --help"
echo "    laax-agent --interactive    (for live demo / evaluation)"
echo "    laax-benchmark              (for overhead profiling)"
echo "    laax-broker                 (for distributed TCP bus)"
echo "======================================================================"
