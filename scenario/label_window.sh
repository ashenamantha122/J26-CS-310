#!/usr/bin/env bash
# Record a labelled time window around one command, for ROC calibration.
#
#   bash scenario/label_window.sh LABELS.csv LABEL -- COMMAND [ARGS...]
#
# Appends "start,end,label,command" (epoch seconds) to LABELS.csv, creating
# the header on first use, and runs COMMAND in between. Use it to wrap each
# Atomic Red Team test you run while the agent is in --observe mode, e.g.
#
#   bash scenario/label_window.sh labels.csv T1059.004 -- \
#       pwsh -NoProfile -Command "Invoke-AtomicTest T1059.004 -TestNumbers 1"
#
# Everything outside the recorded windows (after the agent's learning window)
# counts as benign, so keep other unusual activity off the host meanwhile.
set -uo pipefail

[ $# -ge 4 ] && [ "$3" = "--" ] || {
    echo "usage: $0 LABELS.csv LABEL -- COMMAND [ARGS...]" >&2; exit 2; }
LABELS="$1"; LABEL="$2"; shift 3

case "$LABEL" in *,*|*\"*) echo "label must not contain commas or quotes" >&2; exit 2;; esac
[ -s "$LABELS" ] || echo "start,end,label,command" > "$LABELS"

start="$(date +%s.%N)"
"$@"
rc=$?
end="$(date +%s.%N)"
cmd="$(printf '%s ' "$@" | tr -d ',"' | cut -c1-200)"
echo "${start},${end},${LABEL},\"${cmd% }\"" >> "$LABELS"
echo "labelled ${LABEL}: $(echo "$end - $start" | bc) s, exit ${rc}" >&2
exit $rc
