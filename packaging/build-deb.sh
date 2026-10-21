#!/usr/bin/env bash
# Build the laax-agent Debian package. Run on the target distribution
# (Ubuntu 24.04 / Debian 12+), from anywhere:
#
#   bash packaging/build-deb.sh            -> dist/laax-agent_<ver>_<arch>.deb
#
# Needs: clang, llvm, libbpf-dev, bpftool, make, dpkg-dev (ebpf/setup.sh
# installs the toolchain). The loader is rebuilt from source every time, so
# the package never ships a stale binary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$ROOT/laax_agent.py")"
ARCH="$(dpkg --print-architecture)"
PKG="laax-agent_${VERSION}_${ARCH}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
OUT="$ROOT/dist"

echo "== building loader"
make -C "$ROOT/ebpf" clean >/dev/null
make -C "$ROOT/ebpf" >/dev/null

echo "== staging $PKG"
D="$STAGE/$PKG"
install -d "$D/DEBIAN" "$D/usr/bin" "$D/usr/lib/laax/laax" "$D/usr/lib/laax/services" \
           "$D/etc/laax" "$D/etc/logrotate.d" "$D/lib/systemd/system" \
           "$D/usr/share/doc/laax-agent"

# Python: the agent, the broker and only the modules Component 1 needs. The
# C2/C3/C4 stand-ins are not part of the agent and pull in torch/sklearn.
install -m 0644 "$ROOT/laax_agent.py" "$ROOT/run_broker.py" "$D/usr/lib/laax/"
for f in __init__ backend bus contracts pbcodec zbus laax_pb2; do
    install -m 0644 "$ROOT/laax/$f.py" "$D/usr/lib/laax/laax/"
done
for f in __init__ c1_agent c1_audit c1_preflight c1_scorelog c1_telemetry ebpf_source; do
    install -m 0644 "$ROOT/services/$f.py" "$D/usr/lib/laax/services/"
done
install -m 0755 "$ROOT/ebpf/laax-loader" "$D/usr/lib/laax/laax-loader"

install -m 0755 "$ROOT/packaging/laax-agent.sh" "$D/usr/bin/laax-agent"
install -m 0755 "$ROOT/packaging/laax-broker.sh" "$D/usr/bin/laax-broker"
install -m 0644 "$ROOT/packaging/agent.conf" "$D/etc/laax/agent.conf"
install -m 0644 "$ROOT/packaging/logrotate" "$D/etc/logrotate.d/laax"
install -m 0644 "$ROOT/packaging/laax-agent.service" "$ROOT/packaging/laax-broker.service" \
                "$D/lib/systemd/system/"
install -m 0644 "$ROOT/README.md" "$ROOT/docs/INTEGRATION.md" "$D/usr/share/doc/laax-agent/"
install -m 0644 "$ROOT/LICENSE" "$D/usr/share/doc/laax-agent/copyright"

for f in conffiles; do install -m 0644 "$ROOT/packaging/debian/$f" "$D/DEBIAN/$f"; done
for f in postinst prerm postrm; do install -m 0755 "$ROOT/packaging/debian/$f" "$D/DEBIAN/$f"; done
SIZE="$(du -sk --exclude=DEBIAN "$D" | cut -f1)"
sed -e "s/@VERSION@/$VERSION/" -e "s/@ARCH@/$ARCH/" -e "s/@SIZE@/$SIZE/" \
    "$ROOT/packaging/debian/control" > "$D/DEBIAN/control"

# Files edited on Windows can carry CRLF line endings; a CRLF shebang line
# makes a maintainer script fail with "not found". Normalise everything text.
find "$D" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.service' -o -name '*.conf' \
    -o -path '*/DEBIAN/*' -o -path '*/usr/bin/*' -o -path '*/logrotate.d/*' \) \
    -exec sed -i 's/\r$//' {} +

mkdir -p "$OUT"
dpkg-deb --build --root-owner-group "$D" "$OUT/$PKG.deb" >/dev/null
echo "== built $OUT/$PKG.deb"
dpkg-deb --info "$OUT/$PKG.deb" | sed -n '/Package:/,/Depends:/p'
echo "Install: sudo apt install $OUT/$PKG.deb"
