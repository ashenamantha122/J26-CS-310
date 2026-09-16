#!/usr/bin/env bash
cd "$(dirname "$0")"
rm -f /tmp/lo /tmp/le
mkfifo /tmp/ctl 2>/dev/null
exec 9<>/tmp/ctl
./laax-loader --lsm <&9 > /tmp/lo 2>/tmp/le &
LP=$!
sleep 2
echo "-- baseline --"; echo "STATS" >&9; sleep 1
# generate several execs
/bin/true; /bin/true; ls >/dev/null
sleep 1; echo "STATS" >&9; sleep 1
kill $LP 2>/dev/null; wait $LP 2>/dev/null
echo "--- stdout ---"; grep -E "READY|LSM|DROP" /tmp/lo
echo "--- stderr ---"; cat /tmp/le
rm -f /tmp/ctl
