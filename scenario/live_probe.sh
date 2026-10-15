#!/usr/bin/env bash
# Drives the live run: ordinary work during the baseline window, then a process
# lineage the baseline has never seen, so the agent has something to react to.
#
#   bash scenario/live_probe.sh <baseline_seconds>
set -uo pipefail
BASELINE="${1:-12}"

# --- benign: the same handful of lineages over and over -------------------
end=$(( $(date +%s) + BASELINE ))
while [ "$(date +%s)" -lt "$end" ]; do
    ls >/dev/null 2>&1
    date >/dev/null 2>&1
    cat /etc/hostname >/dev/null 2>&1
    sleep 0.2
done

sleep 1

# --- anomalous: lineages absent from the baseline profile -----------------
# A shell spawning discovery tools and a listener is the shape of a pivot.
bash -c 'whoami; uname -a; id' >/dev/null 2>&1
bash -c 'nohup timeout 3 python3 -c "import socket,time; s=socket.socket(); s.bind((\"127.0.0.1\",19999)); s.listen(1); time.sleep(3)" >/dev/null 2>&1' >/dev/null 2>&1
bash -c 'chmod u+s /tmp/.laax_probe 2>/dev/null; true' >/dev/null 2>&1
for _ in 1 2 3 4 5; do
    bash -c 'od -N4 /dev/urandom' >/dev/null 2>&1
    sleep 0.2
done
