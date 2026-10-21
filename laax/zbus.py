"""
Protocol Buffers over ZeroMQ: the production transport named in the proposal
(FR5), with CURVE encryption and key-based client authentication (NFR4).

Topology
--------
One forwarding proxy (ZmqBroker), two endpoints:

    publishers  --PUB connect-->  XSUB  [proxy]  XPUB  <--SUB connect--  subscribers

Every component holds one ZmqBus, which owns a PUB and a SUB socket. Frames
are [topic, protobuf bytes] (laax/pbcodec.py). ZeroMQ subscriptions match by
prefix, so the receiver also checks the topic exactly.

Security
--------
With CURVE configured, both proxy sockets are CURVE servers. A client must
know the server's public key (so it cannot be pointed at an impostor proxy),
and its own public key must be in the broker's allow-list directory (the ZAP
handler rejects everyone else before any message flows). Traffic is encrypted
end to end between each client and the proxy. Unlike the TcpBus token, a
stolen frame cannot be replayed onto another connection, and no secret is ever
sent on the wire.

Delivery semantics
------------------
PUB/SUB drops messages for subscribers that are not connected yet ("slow
joiner"). ZmqBus.start() waits for the connections to settle before returning;
call it before the first publish. Telemetry is best-effort by design (the
event-loss metric covers the kernel side); control messages from C3/C4 are
idempotent and lease-bounded, so a lost one fails safe.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import zmq
import zmq.asyncio
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator

from .bus import Bus, Handler
from .contracts import Message
from .pbcodec import CodecError, decode, encode


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------
def generate_keypair(directory: str, name: str) -> tuple[Path, Path]:
    """Write <name>.key (public, shareable) and <name>.key_secret (0600)."""
    os.makedirs(directory, exist_ok=True)
    public, secret = zmq.auth.create_certificates(directory, name)
    os.chmod(secret, 0o600)
    return Path(public), Path(secret)


def load_public(path: str) -> bytes:
    public, _ = zmq.auth.load_certificate(path)
    return public


def load_secret(path: str) -> tuple[bytes, bytes]:
    st = os.stat(path)
    # Owner-only, except systemd's credential store: LoadCredential= copies
    # are 0440 inside a per-service directory that nothing else can enter.
    forbidden = 0o027 if os.path.realpath(path).startswith("/run/credentials/") else 0o077
    if st.st_mode & forbidden:
        raise PermissionError(f"{path} is readable by others (mode {st.st_mode & 0o777:o}); chmod 600 it")
    public, secret = zmq.auth.load_certificate(path)
    if secret is None:
        raise ValueError(f"{path} holds no secret key")
    return public, secret


@dataclass
class CurveClient:
    """What a component needs: its own keypair and the broker's public key."""
    public: bytes
    secret: bytes
    server_public: bytes

    @classmethod
    def from_files(cls, secret_file: str, server_public_file: str) -> "CurveClient":
        public, secret = load_secret(secret_file)
        return cls(public, secret, load_public(server_public_file))

    def apply(self, sock: zmq.Socket) -> None:
        sock.curve_publickey = self.public
        sock.curve_secretkey = self.secret
        sock.curve_serverkey = self.server_public


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class ZmqBus(Bus):
    def __init__(self, pub_endpoint: str = "tcp://127.0.0.1:5600",
                 sub_endpoint: str = "tcp://127.0.0.1:5601",
                 curve: Optional[CurveClient] = None,
                 settle_s: float = 0.3) -> None:
        self.pub_endpoint, self.sub_endpoint = pub_endpoint, sub_endpoint
        self.curve, self.settle_s = curve, settle_s
        self._subs: Dict[str, List[Handler]] = {}
        self._ctx: Optional[zmq.asyncio.Context] = None
        self._pub = self._sub = self._pub_sync = None
        self.published = 0
        self.dropped_frames = 0              # undecodable frames, counted not fatal

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs.setdefault(topic, []).append(handler)
        if self._sub is not None:
            self._sub.setsockopt(zmq.SUBSCRIBE, topic.encode())

    def _socket(self, kind: int):
        assert self._ctx is not None
        sock = self._ctx.socket(kind)
        sock.linger = 0
        if self.curve:
            self.curve.apply(sock)
        return sock

    async def start(self) -> None:
        if self._ctx is not None:
            return
        self._ctx = zmq.asyncio.Context()
        self._pub = self._socket(zmq.PUB)
        self._pub.sndhwm = 1_000_000
        self._pub.connect(self.pub_endpoint)
        # A PUB socket never blocks on send (it drops at the high-water mark),
        # so the hot path uses a synchronous view of the same socket and skips
        # creating an asyncio future per message.
        self._pub_sync = zmq.Socket.shadow(self._pub.underlying)
        self._sub = self._socket(zmq.SUB)
        for topic in self._subs:
            self._sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self._sub.rcvhwm = 1_000_000
        self._sub.connect(self.sub_endpoint)
        await asyncio.sleep(self.settle_s)   # let the CURVE handshakes and subscriptions land

    async def publish(self, topic: str, msg: Message) -> None:
        if self._ctx is None:
            await self.start()
        self._pub_sync.send_multipart([topic.encode(), encode(msg)])
        self.published += 1

    async def run(self) -> None:
        await self.start()
        while True:
            frames = await self._sub.recv_multipart()
            if len(frames) != 2:
                self.dropped_frames += 1
                continue
            topic = frames[0].decode(errors="replace")
            handlers = self._subs.get(topic)
            if not handlers:
                continue                     # prefix match on someone else's topic
            try:
                msg = decode(topic, frames[1])
            except CodecError as exc:
                self.dropped_frames += 1
                print(f"[zbus] dropped frame on {topic}: {exc}")
                continue
            for handler in handlers:
                try:
                    await handler(topic, msg)
                except Exception as exc:     # a bad subscriber must not kill the bus
                    print(f"[zbus] handler error on {topic}: {exc!r}")

    def close(self) -> None:
        for sock in (self._pub, self._sub):
            if sock is not None:
                sock.close(linger=0)
        if self._ctx is not None:
            self._ctx.term()
        self._ctx = self._pub = self._sub = self._pub_sync = None


# --------------------------------------------------------------------------
# Broker
# --------------------------------------------------------------------------
class ZmqBroker:
    """
    XSUB/XPUB forwarding proxy. With server_secret_file and clients_dir it
    accepts only clients whose public key file is in clients_dir.
    """

    def __init__(self, xsub_endpoint: str = "tcp://127.0.0.1:5600",
                 xpub_endpoint: str = "tcp://127.0.0.1:5601",
                 server_secret_file: Optional[str] = None,
                 clients_dir: Optional[str] = None) -> None:
        if bool(server_secret_file) != bool(clients_dir):
            raise ValueError("CURVE needs both the server secret and a clients directory")
        self.xsub_endpoint, self.xpub_endpoint = xsub_endpoint, xpub_endpoint
        self.server_secret_file, self.clients_dir = server_secret_file, clients_dir
        self._ctx = zmq.Context()
        self._thread: Optional[threading.Thread] = None
        self._auth: Optional[ThreadAuthenticator] = None
        self.bound = threading.Event()
        self._stop = threading.Event()
        self.forwarded = 0

    @property
    def secure(self) -> bool:
        return bool(self.server_secret_file)

    def _server_socket(self, kind: int) -> zmq.Socket:
        sock = self._ctx.socket(kind)
        sock.linger = 0
        if self.secure:
            public, secret = load_secret(self.server_secret_file)
            sock.curve_server = True
            sock.curve_publickey = public
            sock.curve_secretkey = secret
        return sock

    def serve_forever(self, on_bound: Optional[Callable[[], None]] = None) -> None:
        """Blocking. Returns after stop() is called from another thread."""
        if self.secure:
            self._auth = ThreadAuthenticator(self._ctx)
            self._auth.start()
            # Only public keys present in clients_dir may connect.
            self._auth.configure_curve(domain="*", location=self.clients_dir)
        xsub = self._server_socket(zmq.XSUB)
        xpub = self._server_socket(zmq.XPUB)
        for sock in (xsub, xpub):
            sock.sndhwm = sock.rcvhwm = 1_000_000
        xsub.bind(self.xsub_endpoint)
        xpub.bind(self.xpub_endpoint)
        self.bound.set()
        if on_bound:
            on_bound()
        # A forwarding loop of our own rather than zmq.proxy_steerable: the
        # latter intermittently ignored TERMINATE in testing, leaving a thread
        # that blocked process exit. Here stop() is a flag checked every 100 ms.
        poller = zmq.Poller()
        poller.register(xsub, zmq.POLLIN)
        poller.register(xpub, zmq.POLLIN)
        routes = {xsub: xpub, xpub: xsub}      # messages one way, subscriptions back
        try:
            while not self._stop.is_set():
                for sock, _ in poller.poll(100):
                    out = routes[sock]
                    for _ in range(1000):      # bounded, so neither direction starves
                        try:
                            frames = sock.recv_multipart(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        out.send_multipart(frames)
                        if sock is xsub:
                            self.forwarded += 1
        except KeyboardInterrupt:
            pass
        finally:
            for sock in (xsub, xpub):
                sock.close(linger=0)
            if self._auth:
                self._auth.stop()

    def start_thread(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        if not self.bound.wait(5.0):
            raise RuntimeError("broker failed to bind")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(5.0)
        self._ctx.term()
