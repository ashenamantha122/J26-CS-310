#!/usr/bin/env bash
cd "$(dirname "$0")"
./laax-loader > /tmp/ns.out 2>/dev/null & LP=$!
sleep 2
bash -c 'echo "LOCAL_PID=$$"; exec /bin/sleep 0.05'
sleep 1
kill $LP 2>/dev/null; wait $LP 2>/dev/null
echo "total events captured: $(grep -c '{' /tmp/ns.out)"
echo "--- last few events ---"
grep '{' /tmp/ns.out | tail -5
