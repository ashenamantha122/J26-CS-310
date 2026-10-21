"""
LAAX Framework: standalone message broker.
Student: Amantha M.A (IT23184312) | Research Project: J26-CS-310

Lets C1, C2, C3 and C4 run as separate processes or on separate hosts.

Two transports:

  zmq (production, proposal FR5): Protocol Buffers over ZeroMQ, XSUB/XPUB
      proxy. With --curve-dir, every connection is CURVE-encrypted and only
      clients whose public key is in <curve-dir>/clients/ may connect.
  tcp (development): JSON lines over TCP with a shared token.

Usage:
    python run_broker.py --transport zmq --curve-dir /etc/laax/curve
    python run_broker.py --transport zmq --insecure                 (development only)
    LAAX_BUS_TOKEN=<secret> python run_broker.py --transport tcp
"""

import argparse
import asyncio
import os
import sys

from laax.bus import Broker


def _zmq_main(args) -> int:
    from laax.zbus import ZmqBroker

    curve = None
    if args.curve_dir:
        # Under systemd (DynamicUser) the 0600 secret arrives via
        # LoadCredential, and $LAAX_CURVE_SECRET points at that copy.
        secret = os.environ.get("LAAX_CURVE_SECRET") or os.path.join(args.curve_dir, "server.key_secret")
        curve = {"server_secret_file": secret,
                 "clients_dir": os.path.join(args.curve_dir, "clients")}
        for path in curve.values():
            if not os.path.exists(path):
                print(f"missing {path} -- create keys with: laax-agent keygen {args.curve_dir}")
                return 2
    broker = ZmqBroker(args.xsub, args.xpub, **(curve or {}))

    def banner() -> None:
        print("=" * 64)
        print("  LAAX broker: Protocol Buffers over ZeroMQ")
        print(f"  publishers connect to   {args.xsub}")
        print(f"  subscribers connect to  {args.xpub}")
        auth = (f"CURVE, clients allowed from {curve['clients_dir']}" if curve
                else "NONE (--insecure)")
        print(f"  Authentication: {auth}")
        print("=" * 64, flush=True)

    try:
        broker.serve_forever(on_bound=banner)
    except KeyboardInterrupt:
        pass
    print("\n[*] LAAX broker stopped.")
    return 0


async def _tcp_main(args) -> int:
    broker = Broker(host=args.host, port=args.port, token=args.token)
    server = await asyncio.start_server(broker._client, broker.host, broker.port)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print("=" * 60)
    print(f"  LAAX Distributed TCP Broker Listening on {addrs}")
    print("  Endpoints ready for C1, C2, C3, C4 component connections")
    print(f"  Authentication: {'token required' if args.token else 'NONE (--insecure)'}")
    print("  Press Ctrl+C to terminate broker")
    print("=" * 60)
    try:
        async with server:
            await server.serve_forever()
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        server.close()
        await server.wait_closed()
        print("\n[*] LAAX Broker stopped cleanly.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LAAX message broker")
    p.add_argument("--transport", choices=["zmq", "tcp"], default="zmq")
    p.add_argument("--xsub", default="tcp://127.0.0.1:5600",
                   help="zmq: endpoint publishers connect to")
    p.add_argument("--xpub", default="tcp://127.0.0.1:5601",
                   help="zmq: endpoint subscribers connect to")
    p.add_argument("--curve-dir", help="zmq: directory with server.key_secret and clients/")
    p.add_argument("--host", default="127.0.0.1", help="tcp: bind host")
    p.add_argument("--port", type=int, default=5599, help="tcp: bind port")
    p.add_argument("--token", default=os.environ.get("LAAX_BUS_TOKEN"),
                   help="tcp: shared secret every client must present (default: $LAAX_BUS_TOKEN)")
    p.add_argument("--insecure", action="store_true",
                   help="allow running without authentication (development only)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    authenticated = args.curve_dir if args.transport == "zmq" else args.token
    if not authenticated and not args.insecure:
        how = "--curve-dir" if args.transport == "zmq" else "LAAX_BUS_TOKEN or --token"
        print("refusing to start without authentication: any local user could publish "
              f"enforcement requests.\nUse {how}, or pass --insecure for development.")
        return 2
    if args.transport == "zmq":
        return _zmq_main(args)
    return asyncio.run(_tcp_main(args))


def cli_entry() -> None:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    cli_entry()
