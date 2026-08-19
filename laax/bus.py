"""
LAAX message bus.

Two interchangeable backends behind one interface:

  InProcBus  -- asyncio queues, zero dependencies. The default, so the whole
                framework runs on any machine with a stock Python.
  TcpBus     -- newline-delimited JSON over TCP, one broker process. Use when
                components move to separate hosts or separate teams.

The production target is Protocol Buffers over ZeroMQ (proposal slide 7).
Swapping it in means writing a third backend with this same three-method
surface -- publish / subscribe / run -- and changing one line in run_demo.py.
No component code changes, because components only ever touch Bus.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, List

from .contracts import Message, decode

Handler = Callable[[str, Message], Awaitable[None]]


class Bus:
    """Interface every backend implements."""

    async def publish(self, topic: str, msg: Message) -> None:
        raise NotImplementedError

    def subscribe(self, topic: str, handler: Handler) -> None:
        raise NotImplementedError

    async def run(self) -> None:
        raise NotImplementedError


class InProcBus(Bus):
    """Single-process fan-out. Ordering is preserved per topic."""

    def __init__(self, trace: bool = False) -> None:
        self._subs: Dict[str, List[Handler]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._trace = trace
        self.published = 0

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs.setdefault(topic, []).append(handler)

    async def publish(self, topic: str, msg: Message) -> None:
        # Round-trip through dict form so the in-process path exercises exactly
        # the same serialisation boundary the network backends will.
        await self._queue.put((topic, msg.to_dict()))
        self.published += 1

    async def run(self) -> None:
        while True:
            topic, payload = await self._queue.get()
            if self._trace:
                print(f"[bus] {topic} {json.dumps(payload)[:110]}")
            msg = decode(topic, payload)
            for handler in self._subs.get(topic, []):
                try:
                    await handler(topic, msg)
                except Exception as exc:  # a bad subscriber must not kill the bus
                    print(f"[bus] handler error on {topic}: {exc!r}")
            self._queue.task_done()

    async def drain(self) -> None:
        await self._queue.join()


class TcpBus(Bus):
    """JSON-lines over TCP. Start one Broker, then one TcpBus per component."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5599) -> None:
        self.host, self.port = host, port
        self._subs: Dict[str, List[Handler]] = {}
        self._writer: asyncio.StreamWriter | None = None

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs.setdefault(topic, []).append(handler)

    async def _connect(self) -> None:
        reader, writer = await asyncio.open_connection(self.host, self.port)
        self._writer = writer
        writer.write((json.dumps({"op": "sub", "topics": list(self._subs)}) + "\n").encode())
        await writer.drain()
        self._reader = reader

    async def publish(self, topic: str, msg: Message) -> None:
        if self._writer is None:
            await self._connect()
        line = json.dumps({"op": "pub", "topic": topic, "msg": msg.to_dict()}) + "\n"
        self._writer.write(line.encode())
        await self._writer.drain()

    async def run(self) -> None:
        if self._writer is None:
            await self._connect()
        while True:
            line = await self._reader.readline()
            if not line:
                return
            frame = json.loads(line)
            topic, payload = frame["topic"], frame["msg"]
            msg = decode(topic, payload)
            for handler in self._subs.get(topic, []):
                await handler(topic, msg)


class Broker:
    """Fan-out server for TcpBus."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5599) -> None:
        self.host, self.port = host, port
        self._clients: List[tuple[asyncio.StreamWriter, set]] = []

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        topics: set = set()
        entry = (writer, topics)
        self._clients.append(entry)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                frame = json.loads(line)
                if frame["op"] == "sub":
                    topics.update(frame["topics"])
                elif frame["op"] == "pub":
                    out = (json.dumps({"topic": frame["topic"], "msg": frame["msg"]}) + "\n").encode()
                    for w, t in list(self._clients):
                        if frame["topic"] in t and not w.is_closing():
                            w.write(out)
        finally:
            self._clients.remove(entry)
            writer.close()

    async def run(self) -> None:
        server = await asyncio.start_server(self._client, self.host, self.port)
        async with server:
            await server.serve_forever()
