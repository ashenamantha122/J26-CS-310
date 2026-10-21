#!/usr/bin/env bash
# Drives the live run: ordinary work during the baseline window, then a process
# lineage the baseline has never seen, so the agent has something to react to.
#
#   bash scenario/live_probe.sh <baseline_seconds> [ready_file]
#
# With ready_file, wait until run_live.py --ready-file creates it, so the
# benign phase lines up with the agent's learning window however long the
# agent takes to start.
set -uo pipefail
BASELINE="${1:-12}"
READY="${2:-}"

if [ -n "$READY" ]; then
    for _ in $(seq 1 600); do
        [ -e "$READY" ] && break
        sleep 0.1
    done
    [ -e "$READY" ] || { echo "live_probe: agent never became ready" >&2; exit 1; }
fi

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

# --- a suspicious shell that keeps working ---------------------------------
# Everything above lives for milliseconds: each command has finished before
# the agent's decision reaches the kernel, so there is nothing left to trace
# deeply. An attacker's shell is not like that -- it stays and keeps running
# commands. Once it is promoted, every later child is captured with full argv,
# in-kernel, from the fork onwards. This is the case deep mode exists for.
sleep 1
bash -c 'whoami; sleep 0.5; id -un; cat /etc/hostname; ls -la /tmp; uname -r;
         sleep 0.5; head -c 32 /dev/urandom | base64; ps -o pid,comm' >/dev/null 2>&1
