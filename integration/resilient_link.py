"""
resilient_link.py — the "cannot crash" layer between Component 4 and the
rest of LAAX (Component 1, Component 3).

Why this file exists:
Components 1-3 are other people's code, at various stages of completion.
Component 4 is the only component that takes real-world action, so it is
the one place a silent hang, an unhandled exception, or a malformed
message from someone else's component must NEVER be allowed to kill the
process. Every upstream/downstream link goes through this module instead
of raw ZeroMQ calls scattered through the codebase.

Three protections, each independently testable:
  1. Heartbeat + timeout  -> detect a dead/slow peer instead of blocking forever
  2. Schema validation    -> reject malformed messages without raising
  3. Circuit breaker      -> stop hammering a peer that keeps failing,
                             and let the rest of the system keep running
                             in a degraded-but-alive state

Dependencies: pip install pyzmq jsonschema
(Swap in real protobuf parsing later with the same validate() contract —
see messages.proto. JSON is used here because it matches your existing
stub feed and demos without a toolchain change.)
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import zmq
from jsonschema import validate as _jsonschema_validate, ValidationError

logger = logging.getLogger("component4.integration")


# ---------------------------------------------------------------------------
# Message schemas (JSON mirror of messages.proto — keep these in sync)
# ---------------------------------------------------------------------------

THREAT_ALERT_SCHEMA = {
    "type": "object",
    "required": ["alert_id", "timestamp_ms", "threat_type", "confidence",
                 "asset_id", "asset_type", "asset_criticality"],
    "properties": {
        "alert_id": {"type": "string", "minLength": 1},
        "timestamp_ms": {"type": "number"},
        "threat_type": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "asset_id": {"type": "string", "minLength": 1},
        "asset_type": {"type": "string", "enum": ["MANAGED_LINUX", "IOT_AGENTLESS", "ASSET_UNKNOWN"]},
        "asset_criticality": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL", "CRITICALITY_UNKNOWN"]},
        "source_component": {"type": "string"},
        "evidence_summary": {"type": "string"},
        "related_alert_ids": {"type": "array", "items": {"type": "string"}},
    },
}

LSM_COMMAND_ACK_SCHEMA = {
    "type": "object",
    "required": ["command_id", "applied"],
    "properties": {
        "command_id": {"type": "string"},
        "applied": {"type": "boolean"},
        "error": {"type": "string"},
        "applied_at_ms": {"type": "number"},
    },
}


def validate_message(payload: dict, schema: dict) -> Optional[str]:
    """Returns None if valid, else a short error string. Never raises."""
    try:
        _jsonschema_validate(instance=payload, schema=schema)
        return None
    except ValidationError as e:
        return f"schema violation: {e.message}"
    except Exception as e:  # noqa: BLE001 - deliberately broad, this must never propagate
        return f"unexpected validation error: {e}"


# ---------------------------------------------------------------------------
# Circuit breaker — per upstream peer (e.g. "component3", "component1")
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = "closed"        # normal operation
    OPEN = "open"            # peer is failing, stop calling it
    HALF_OPEN = "half_open"  # probing whether peer has recovered


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    recovery_timeout_s: float = 15.0
    _state: CircuitState = field(default=CircuitState.CLOSED)
    _failure_count: int = field(default=0)
    _opened_at: float = field(default=0.0)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow_call(self) -> bool:
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                if time.time() - self._opened_at >= self.recovery_timeout_s:
                    self._state = CircuitState.HALF_OPEN
                    logger.warning("circuit[%s] OPEN -> HALF_OPEN, probing", self.name)
                    return True
                return False
            return True  # HALF_OPEN: allow the probe call through

    def record_success(self):
        with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info("circuit[%s] recovered -> CLOSED", self.name)
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                logger.warning("circuit[%s] probe failed -> OPEN again", self.name)
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
                logger.error("circuit[%s] tripped -> OPEN after %d failures",
                             self.name, self._failure_count)

    @property
    def state(self) -> CircuitState:
        return self._state


# ---------------------------------------------------------------------------
# Resilient subscriber: receives ThreatAlerts from Component 3
# ---------------------------------------------------------------------------

class ResilientSubscriber:
    """
    Wraps a ZMQ SUB socket with:
      - a receive timeout (never blocks forever if C3 is down)
      - heartbeat-based liveness tracking
      - schema validation before the callback ever sees a message
      - a circuit breaker so a storm of bad messages doesn't spin-loop

    Usage:
        sub = ResilientSubscriber("component3", "tcp://127.0.0.1:5556",
                                   schema=THREAT_ALERT_SCHEMA,
                                   on_message=handle_alert,
                                   on_peer_down=lambda: decision_engine.enter_degraded_mode(),
                                   on_peer_up=lambda: decision_engine.exit_degraded_mode())
        sub.start()   # runs its own thread; never blocks the caller
        ...
        sub.stop()
    """

    def __init__(self, peer_name: str, connect_addr: str, schema: dict,
                 on_message: Callable[[dict], None],
                 on_peer_down: Optional[Callable[[], None]] = None,
                 on_peer_up: Optional[Callable[[], None]] = None,
                 recv_timeout_ms: int = 1000,
                 heartbeat_timeout_s: float = 10.0,
                 topic: bytes = b""):
        self.peer_name = peer_name
        self.connect_addr = connect_addr
        self.schema = schema
        self.on_message = on_message
        self.on_peer_down = on_peer_down
        self.on_peer_up = on_peer_up
        self.recv_timeout_ms = recv_timeout_ms
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.topic = topic

        self.breaker = CircuitBreaker(name=peer_name)
        self._ctx = zmq.Context.instance()
        self._sock: Optional[zmq.Socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._last_seen = 0.0
        self._peer_alive = False

    def _connect(self):
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001
                pass
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.connect_addr)
        self._sock.setsockopt(zmq.SUBSCRIBE, self.topic)
        logger.info("subscriber[%s] connected to %s", self.peer_name, self.connect_addr)

    def start(self):
        self._connect()
        self._thread = threading.Thread(target=self._run, name=f"sub-{self.peer_name}", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._sock:
            self._sock.close(linger=0)

    def _run(self):
        self._last_seen = time.time()
        while not self._stop_evt.is_set():
            try:
                raw = self._sock.recv(flags=0)
            except zmq.Again:
                # No message within recv_timeout_ms — normal, not an error.
                self._check_liveness()
                continue
            except zmq.ZMQError as e:
                logger.error("subscriber[%s] socket error: %s — reconnecting in 2s", self.peer_name, e)
                time.sleep(2)
                self._safe_reconnect()
                continue
            except Exception as e:  # noqa: BLE001 - absolute last resort guard
                logger.exception("subscriber[%s] unexpected error in recv loop: %s", self.peer_name, e)
                time.sleep(1)
                continue

            self._last_seen = time.time()
            self._mark_peer_up()
            self._handle_raw(raw)

    def _handle_raw(self, raw: bytes):
        if not self.breaker.allow_call():
            logger.debug("subscriber[%s] circuit open — dropping message", self.peer_name)
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("subscriber[%s] non-JSON message dropped: %s", self.peer_name, e)
            self.breaker.record_failure()
            return

        err = validate_message(payload, self.schema)
        if err:
            logger.warning("subscriber[%s] invalid message dropped: %s | payload=%s",
                           self.peer_name, err, str(payload)[:200])
            self.breaker.record_failure()
            return

        try:
            self.on_message(payload)
            self.breaker.record_success()
        except Exception as e:  # noqa: BLE001 - a bug in OUR handler must not kill the recv loop
            logger.exception("subscriber[%s] on_message handler raised: %s", self.peer_name, e)
            self.breaker.record_failure()

    def _check_liveness(self):
        if self._peer_alive and (time.time() - self._last_seen) > self.heartbeat_timeout_s:
            self._peer_alive = False
            logger.warning("subscriber[%s] peer silent for >%.0fs — marking DOWN",
                           self.peer_name, self.heartbeat_timeout_s)
            if self.on_peer_down:
                try:
                    self.on_peer_down()
                except Exception:  # noqa: BLE001
                    logger.exception("subscriber[%s] on_peer_down callback failed", self.peer_name)

    def _mark_peer_up(self):
        if not self._peer_alive:
            self._peer_alive = True
            logger.info("subscriber[%s] peer UP", self.peer_name)
            if self.on_peer_up:
                try:
                    self.on_peer_up()
                except Exception:  # noqa: BLE001
                    logger.exception("subscriber[%s] on_peer_up callback failed", self.peer_name)

    def _safe_reconnect(self):
        try:
            self._connect()
        except Exception:  # noqa: BLE001
            logger.exception("subscriber[%s] reconnect failed, will retry", self.peer_name)

    @property
    def is_peer_alive(self) -> bool:
        return self._peer_alive


# ---------------------------------------------------------------------------
# Resilient publisher: sends LsmCommands to Component 1 (or anything downstream)
# ---------------------------------------------------------------------------

class ResilientPublisher:
    """
    Fire-and-forget PUB socket wrapper. Publishing never blocks or raises
    out of send(); failures are logged and counted via the circuit breaker
    so callers (the Decision Engine) can check `publisher.breaker.state`
    before deciding whether to also fall back to a local-only action.
    """

    def __init__(self, peer_name: str, bind_addr: str):
        self.peer_name = peer_name
        self.bind_addr = bind_addr
        self.breaker = CircuitBreaker(name=f"{peer_name}-pub")
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(bind_addr)
        logger.info("publisher[%s] bound on %s", peer_name, bind_addr)

    def send(self, payload: dict, topic: bytes = b"") -> bool:
        if not self.breaker.allow_call():
            logger.debug("publisher[%s] circuit open — send skipped", self.peer_name)
            return False
        try:
            data = json.dumps(payload).encode("utf-8")
            self._sock.send(topic + data, flags=zmq.NOBLOCK)
            self.breaker.record_success()
            return True
        except Exception as e:  # noqa: BLE001
            logger.exception("publisher[%s] send failed: %s", self.peer_name, e)
            self.breaker.record_failure()
            return False

    def close(self):
        self._sock.close(linger=0)


def new_alert_id() -> str:
    return str(uuid.uuid4())
