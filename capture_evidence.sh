#!/usr/bin/env bash
# Bank every Linux-side result to demo_evidence/ so a live failure tomorrow
# costs you nothing. Run tonight, from Windows:
#
#   wsl -d Ubuntu-24.04 -u root -- bash "/mnt/e/Accepted RP/Prototype/capture_evidence.sh"
#
# Then capture the two Windows-side runs separately (see DEMO_GUIDE.txt PART 0).

cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
# Absolute, because a stray `cd` in one capture block must not silently
# redirect the next block's output somewhere else.
OUT="$ROOT/demo_evidence"
mkdir -p "$OUT"

mount -t securityfs securityfs /sys/kernel/security 2>/dev/null

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "capturing to $OUT/ ..."

(
    echo "captured: $(stamp)"
    echo "kernel:   $(uname -r)"
    echo "lsm:      $(cat /sys/kernel/security/lsm 2>/dev/null || echo unavailable)"
    echo
    cd "$ROOT/ebpf" && make check 2>&1
) > "$OUT/kernel_check.txt" 2>&1
echo "  [1/4] kernel_check.txt"

# Make sure the loader is current before anything depends on it.
( cd "$ROOT/ebpf" && make ) > "$OUT/build.txt" 2>&1
echo "  [2/4] build.txt"

(
    cd "$ROOT"
    echo "captured: $(stamp)"
    echo "command:  python3 run_live.py --calibrate --iterations 30"
    echo
    python3 run_live.py --calibrate --iterations 30 2>&1
) > "$OUT/calibration.txt"
echo "  [3/4] calibration.txt   <- your headline number"

(
    cd "$ROOT"
    echo "captured: $(stamp)"
    echo "command:  python3 scenario/test_lsm.py"
    echo
    python3 scenario/test_lsm.py 2>&1
) > "$OUT/lsm_test.txt"
echo "  [4/4] lsm_test.txt"

(
    cd "$ROOT"
    echo "captured: $(stamp)"
    echo "command:  python3 scenario/test_inheritance.py"
    echo
    python3 scenario/test_inheritance.py 2>&1
) > "$OUT/inheritance_test.txt"
echo "  [+]   inheritance_test.txt"

rm -f "$OUT/live_audit.jsonl" "$OUT/live_telemetry.csv"
READY="$(mktemp -u /tmp/laax_ready.XXXXXX)"
(
    cd "$ROOT"
    echo "captured: $(stamp)"
    echo "command:  live_probe.sh 12 READY & run_live.py --lsm --baseline 12 --seconds 16" \
         "--ready-file READY --audit demo_evidence/live_audit.jsonl" \
         "--export demo_evidence/live_telemetry.csv"
    echo
    bash scenario/live_probe.sh 12 "$READY" &
    python3 run_live.py --lsm --baseline 12 --seconds 16 --ready-file "$READY" \
        --audit "$OUT/live_audit.jsonl" --export "$OUT/live_telemetry.csv" 2>&1
    rm -f "$READY"
    wait
    echo
    echo "audit chain:  $(python3 -m services.c1_audit verify "$OUT/live_audit.jsonl")"
    echo "C3 schema:    $(python3 -m services.c1_telemetry "$OUT/live_telemetry.csv" | head -3)"
) > "$OUT/live_agent.txt"
echo "  [+]   live_agent.txt (+ live_audit.jsonl, live_telemetry.csv)"

(
    cd "$ROOT"
    echo "captured: $(stamp)"
    echo "command:  python3 benchmarks/kernel_overhead.py --runs 10"
    echo
    python3 benchmarks/kernel_overhead.py --runs 10 2>&1 | grep -vE "run [0-9]+/|^\s+\[C1\]"
) > "$OUT/kernel_overhead.txt"
echo "  [+]   kernel_overhead.txt"

(
    cd "$ROOT"
    echo "captured: $(stamp)"
    for c in "" "--no-curve"; do
        echo; echo "command:  python3 benchmarks/transport_throughput.py --events 100000 $c"
        python3 benchmarks/transport_throughput.py --events 100000 $c 2>&1
    done
) > "$OUT/transport_throughput.txt"
echo "  [+]   transport_throughput.txt"

(
    cd "$ROOT"
    echo "captured: $(stamp)"
    echo "command:  python3 -m unittest discover -s tests -v"
    echo
    python3 -m unittest discover -s tests -v 2>&1 | grep -vE "^\s+\[C1\]"
) > "$OUT/unit_tests.txt"
echo "  [+]   unit_tests.txt"

if [ "${LAAX_TEST_INSTALL:-0}" = 1 ]; then
    # Installs, runs and PURGES the package on this machine -- opt-in only.
    ( echo "captured: $(stamp)"; bash "$ROOT/packaging/test-install.sh" 2>&1 ) > "$OUT/package_install.txt"
    echo "  [+]   package_install.txt"
fi

echo
echo "done. Files in $OUT:"
ls -la "$OUT"
echo
echo "Now capture the Windows-side runs from PowerShell:"
echo '  python run_demo.py           > demo_evidence\windows_demo.txt 2>&1'
echo '  python run_demo.py --c2-eval > demo_evidence\c2_eval.txt      2>&1'
echo
echo "Then OPEN each file and confirm it is not empty."
