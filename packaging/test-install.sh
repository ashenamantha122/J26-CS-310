#!/usr/bin/env bash
# End-to-end test of the Debian package on this machine, as root:
#
#   sudo bash packaging/test-install.sh        (builds dist/*.deb first)
#
# install -> check -> start -> status -> log rotation -> CURVE broker + agent
# over ZeroMQ with a C3 stand-in subscriber -> stop -> purge.
#
# WARNING: installs, starts and finally PURGES laax-agent on this host, and
# creates throwaway CURVE keys. Run it on a test machine or WSL, not on a host
# whose LAAX installation you want to keep.
set -uo pipefail
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pass=0; fail=0
ok()   { echo "  PASS  $*"; pass=$((pass + 1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

echo "== build"
bash "$ROOT/packaging/build-deb.sh" >/dev/null 2>&1 || { echo "build failed"; exit 1; }
DEB="$(ls "$ROOT"/dist/laax-agent_*.deb | tail -1)"
cp "$DEB" /tmp/laax-test.deb                 # apt cannot read from some mounts

echo "== install"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq /tmp/laax-test.deb >/dev/null 2>&1
check "package installed" '[ "$(dpkg-query -W -f="\${Status}" laax-agent 2>/dev/null)" = "install ok installed" ]'
check "laax-agent on PATH" 'command -v laax-agent >/dev/null'
check "preflight passes" 'laax-agent check --config /etc/laax/agent.conf >/dev/null'

echo "== service"
systemctl start laax-agent; sleep 8
check "service active" '[ "$(systemctl is-active laax-agent)" = active ]'
for _ in $(seq 1 10); do ls / >/dev/null; sleep 0.2; done; sleep 6
check "status reports running" 'laax-agent status >/dev/null'
check "status readable unprivileged" 'su nobody -s /bin/sh -c "laax-agent status" >/dev/null'
check "telemetry written" '[ "$(wc -l < /var/log/laax/telemetry.csv)" -gt 1 ]'

echo "== log rotation"
logrotate -f /etc/logrotate.d/laax; sleep 2
for _ in $(seq 1 10); do id >/dev/null; sleep 0.2; done; sleep 3
check "rotated file exists" '[ -e /var/log/laax/telemetry.csv.1 ]'
check "new file starts with the C3 header" 'head -1 /var/log/laax/telemetry.csv | grep -q "^timestamp,host_id"'
check "new file receives rows" '[ "$(wc -l < /var/log/laax/telemetry.csv)" -gt 1 ]'

echo "== CURVE broker + agent over ZeroMQ"
K=/etc/laax/curve
laax-agent keygen $K --server >/dev/null
laax-agent keygen $K --name agent >/dev/null && mv $K/agent.key $K/clients/
rm -rf /tmp/laax-c3keys
laax-agent keygen /tmp/laax-c3keys --name c3 >/dev/null && cp /tmp/laax-c3keys/c3.key $K/clients/
mkdir -p /etc/systemd/system/laax-agent.service.d
cat > /etc/systemd/system/laax-agent.service.d/zmq.conf <<'X'
[Service]
ExecStart=
ExecStart=/usr/bin/laax-agent run --config /etc/laax/agent.conf --bus zmq --curve-key /etc/laax/curve/agent.key_secret --curve-server-key /etc/laax/curve/server.key --baseline 5
X
systemctl daemon-reload
systemctl stop laax-agent
systemctl start laax-broker; sleep 2
check "broker active" '[ "$(systemctl is-active laax-broker)" = active ]'
cat > /tmp/laax-c3sub.py <<'X'
import asyncio, sys
sys.path.insert(0, "/usr/lib/laax")
from laax.zbus import ZmqBus, CurveClient
from laax.contracts import Topic
async def main():
    bus = ZmqBus(curve=CurveClient.from_files(sys.argv[1], "/etc/laax/curve/server.key"))
    got = []
    async def h(t, m): got.append(m)
    bus.subscribe(Topic.HOST_EVENT, h); await bus.start()
    t = asyncio.create_task(bus.run()); await asyncio.sleep(12); t.cancel(); bus.close()
    print(len(got))
asyncio.run(main())
X
python3 /tmp/laax-c3sub.py /tmp/laax-c3keys/c3.key_secret > /tmp/laax-c3.out &
SUB=$!
rm -rf /tmp/laax-intruder; laax-agent keygen /tmp/laax-intruder --name intruder >/dev/null
python3 /tmp/laax-c3sub.py /tmp/laax-intruder/intruder.key_secret > /tmp/laax-intruder.out &
BAD=$!
sleep 1; systemctl start laax-agent; sleep 3
for _ in $(seq 1 20); do ls /tmp >/dev/null; sleep 0.3; done
wait $SUB $BAD
check "agent on CURVE bus" 'journalctl -u laax-agent -n 40 -o cat | grep -q "auth CURVE"'
check "allowed subscriber received telemetry ($(cat /tmp/laax-c3.out) events)" '[ "$(cat /tmp/laax-c3.out)" -gt 0 ]'
check "unlisted key received nothing ($(cat /tmp/laax-intruder.out) events)" '[ "$(cat /tmp/laax-intruder.out)" = 0 ]'

echo "== stop"
t0=$(date +%s%N); systemctl stop laax-agent laax-broker; ms=$(( ($(date +%s%N) - t0) / 1000000 ))
check "clean stop in ${ms} ms" '[ "$(systemctl is-active laax-agent)" = inactive ]'
check "audit chain intact" 'python3 -sE /usr/lib/laax/laax_agent.py verify-audit /var/log/laax/audit.jsonl >/dev/null'
rm -rf /etc/systemd/system/laax-agent.service.d; systemctl daemon-reload

echo "== purge"
DEBIAN_FRONTEND=noninteractive apt-get purge -y -qq laax-agent >/dev/null 2>&1
for p in /usr/bin/laax-agent /usr/lib/laax /etc/laax /lib/systemd/system/laax-agent.service /run/laax; do
    check "removed $p" "[ ! -e $p ]"
done
check "logs kept for the operator" '[ -d /var/log/laax ]'
rm -rf /tmp/laax-c3keys /tmp/laax-intruder /tmp/laax-c3sub.py /tmp/laax-*.out /tmp/laax-test.deb

echo
echo "  $pass passed, $fail failed"
[ "$fail" = 0 ]
