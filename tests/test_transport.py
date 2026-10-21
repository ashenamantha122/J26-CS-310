"""
Protocol Buffers codec and the ZeroMQ/CURVE transport (FR5, NFR4).
IT23184312 | Amantha M A

Skipped when pyzmq or protobuf is not installed (python3-zmq, python3-protobuf).
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import socket
import stat
import tempfile
import unittest

try:
    import zmq  # noqa: F401
    from laax import pbcodec
    from laax.zbus import CurveClient, ZmqBroker, ZmqBus, generate_keypair, load_secret
    HAVE_ZMQ = True
except ImportError:                       # pragma: no cover - environment dependent
    HAVE_ZMQ = False

from laax.contracts import (DECODERS, CorrelationAlert, EnforceHost, HostEvent,
                            ModeChange, Topic)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(HAVE_ZMQ, "pyzmq/protobuf not installed")
class CodecTests(unittest.TestCase):
    def test_every_contract_field_has_a_proto_field(self):
        for cls in DECODERS.values():
            pb_fields = set(pbcodec._pb_class(cls).DESCRIPTOR.fields_by_name)
            for f in dataclasses.fields(cls):
                name = "path_json" if (cls is CorrelationAlert and f.name == "path") else f.name
                self.assertIn(name, pb_fields, f"{cls.__name__}.{f.name}")

    def test_round_trip_every_topic_with_defaults(self):
        for topic, cls in DECODERS.items():
            msg = cls()
            self.assertEqual(pbcodec.decode(topic, pbcodec.encode(msg)), msg, cls.__name__)

    def test_host_event_round_trip_keeps_none_and_values_apart(self):
        ev = HostEvent(host_id="h", pid=4242, ppid=1, comm="nc", exe="/usr/bin/nc",
                       kind="connect", depth="deep", hook="kprobe",
                       daddr="2001:db8::1", dport=443, sport=51000, proto="tcp",
                       argv=["nc", "two words"], ret=-13, rarity=6.5, confidence=9.25)
        back = pbcodec.decode(Topic.HOST_EVENT, pbcodec.encode(ev))
        self.assertEqual(back, ev)
        empty = HostEvent(argv=[])
        self.assertEqual(pbcodec.decode(Topic.HOST_EVENT, pbcodec.encode(empty)).argv, [])
        none = HostEvent()
        decoded = pbcodec.decode(Topic.HOST_EVENT, pbcodec.encode(none))
        self.assertIsNone(decoded.argv)
        self.assertIsNone(decoded.ret)
        self.assertIsNone(decoded.saddr)

    def test_correlation_path_survives(self):
        alert = CorrelationAlert(path=[{"node": "iot:cam", "type": "iot_device"}],
                                 pid=1400, host_id="h")
        self.assertEqual(pbcodec.decode(Topic.CORRELATION, pbcodec.encode(alert)), alert)

    def test_garbage_is_a_codec_error_not_a_crash(self):
        with self.assertRaises(pbcodec.CodecError):
            pbcodec.decode(Topic.ENFORCE_HOST, b"\xff\xff\xff\xff")
        with self.assertRaises(pbcodec.CodecError):
            pbcodec.decode("no.such.topic", b"")

    def test_protobuf_is_smaller_than_json(self):
        import json
        ev = HostEvent(host_id="host-0", pid=4471, ppid=1400, comm="bash",
                       kind="exec", hook="tracepoint", exe="/usr/bin/bash")
        self.assertLess(len(pbcodec.encode(ev)), len(json.dumps(ev.to_dict())))


@unittest.skipUnless(HAVE_ZMQ, "pyzmq/protobuf not installed")
class ZmqBusTests(unittest.IsolatedAsyncioTestCase):
    def endpoints(self):
        return f"tcp://127.0.0.1:{free_port()}", f"tcp://127.0.0.1:{free_port()}"

    async def deliver(self, pub_curve=None, sub_curve=None, broker_kw=None,
                      timeout: float = 1.5):
        xsub, xpub = self.endpoints()
        broker = ZmqBroker(xsub, xpub, **(broker_kw or {}))
        broker.start_thread()
        got: list = []
        sub = ZmqBus(xsub, xpub, curve=sub_curve)
        pub = ZmqBus(xsub, xpub, curve=pub_curve)

        async def handler(_t, m):
            got.append(m)

        sub.subscribe(Topic.ENFORCE_HOST, handler)
        task = asyncio.create_task(sub.run())
        try:
            await pub.start()
            await asyncio.sleep(0.3)
            await pub.publish(Topic.ENFORCE_HOST, EnforceHost(pid=77, policy_id="p"))
            await pub.publish(Topic.ENFORCE_HOST + ".decoy", EnforceHost(pid=1))
            deadline = asyncio.get_running_loop().time() + timeout
            while not got and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
        finally:
            task.cancel()
            pub.close()
            sub.close()
            broker.stop()
        return got

    async def test_plain_delivery_and_exact_topic_match(self):
        got = await self.deliver()
        self.assertEqual([m.pid for m in got], [77])

    async def curve_setup(self, d):
        _, server_secret = generate_keypair(d, "server")
        clients = os.path.join(d, "clients")
        os.makedirs(clients)
        keys = {}
        for name in ("agent", "c4"):
            pub_file, sec_file = generate_keypair(d, name)
            os.replace(pub_file, os.path.join(clients, f"{name}.key"))
            keys[name] = sec_file
        _, intruder = generate_keypair(os.path.join(d, "other"), "intruder")
        server_pub = os.path.join(d, "server.key")
        mk = lambda sec: CurveClient.from_files(str(sec), server_pub)
        return {"server_secret_file": str(server_secret), "clients_dir": clients}, \
            mk(keys["agent"]), mk(keys["c4"]), mk(intruder)

    async def test_curve_allowed_clients_deliver(self):
        with tempfile.TemporaryDirectory() as d:
            broker_kw, agent, c4, _ = await self.curve_setup(d)
            got = await self.deliver(pub_curve=c4, sub_curve=agent, broker_kw=broker_kw)
            self.assertEqual([m.pid for m in got], [77])

    async def test_curve_unknown_publisher_cannot_command_the_agent(self):
        with tempfile.TemporaryDirectory() as d:
            broker_kw, agent, _, intruder = await self.curve_setup(d)
            got = await self.deliver(pub_curve=intruder, sub_curve=agent,
                                     broker_kw=broker_kw, timeout=1.0)
            self.assertEqual(got, [])

    async def test_plaintext_client_cannot_reach_a_curve_broker(self):
        with tempfile.TemporaryDirectory() as d:
            broker_kw, agent, _, _ = await self.curve_setup(d)
            got = await self.deliver(pub_curve=None, sub_curve=agent,
                                     broker_kw=broker_kw, timeout=1.0)
            self.assertEqual(got, [])

    async def test_mode_change_round_trips_over_the_wire(self):
        xsub, xpub = self.endpoints()
        broker = ZmqBroker(xsub, xpub)
        broker.start_thread()
        sub, pub = ZmqBus(xsub, xpub), ZmqBus(xsub, xpub)
        got = []

        async def handler(_t, m):
            got.append(m)

        sub.subscribe(Topic.HOST_MODE, handler)
        task = asyncio.create_task(sub.run())
        try:
            await pub.start()
            await asyncio.sleep(0.3)
            sent = ModeChange(pid=5, probes=["kprobe/finalize_exec"], confidence=8.5)
            await pub.publish(Topic.HOST_MODE, sent)
            for _ in range(30):
                if got:
                    break
                await asyncio.sleep(0.05)
            self.assertEqual(got, [sent])
        finally:
            task.cancel()
            pub.close()
            sub.close()
            broker.stop()


@unittest.skipUnless(HAVE_ZMQ, "pyzmq/protobuf not installed")
class AgentOverZmqTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_takes_c4_orders_and_feeds_c3_over_curve(self):
        from laax.backend import SimulatedBackend
        from services.c1_agent import C1Agent, ProcState

        with tempfile.TemporaryDirectory() as d:
            broker_kw, agent_keys, c4_keys, _ = await ZmqBusTests.curve_setup(None, d)
            xsub, xpub = f"tcp://127.0.0.1:{free_port()}", f"tcp://127.0.0.1:{free_port()}"
            broker = ZmqBroker(xsub, xpub, **broker_kw)
            broker.start_thread()

            class Src:
                def __init__(self):
                    self.q = asyncio.Queue()

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    return await self.q.get()

            src = Src()
            agent_bus = ZmqBus(xsub, xpub, curve=agent_keys)
            agent = C1Agent(agent_bus, src, host_id="h", backend=SimulatedBackend())
            peer = ZmqBus(xsub, xpub, curve=c4_keys)      # plays C3 and C4
            seen = []

            async def on_event(_t, m):
                seen.append(m)

            peer.subscribe(Topic.HOST_EVENT, on_event)
            tasks = [asyncio.create_task(agent_bus.run()), asyncio.create_task(agent.run()),
                     asyncio.create_task(peer.run())]
            try:
                await agent_bus.start()
                await peer.start()
                await asyncio.sleep(0.3)
                await src.q.put(HostEvent(host_id="h", pid=900, ppid=1, comm="sshd",
                                          kind="exec", hook="tracepoint"))
                for _ in range(40):
                    if seen:
                        break
                    await asyncio.sleep(0.05)
                self.assertEqual([e.pid for e in seen], [900])       # C1 -> C3

                await peer.publish(Topic.ENFORCE_HOST,
                                   EnforceHost(host_id="h", pid=900, policy_id="pol",
                                               lease_seconds=30))
                for _ in range(40):
                    if agent.denied.get(900):
                        break
                    await asyncio.sleep(0.05)
                self.assertIn("execve", agent.denied[900])            # C4 -> C1
                armed = [r for r in agent.audit.records if r["event"] == "enforce_armed"]
                self.assertEqual(armed[0]["policy_id"], "pol")
            finally:
                for t in tasks:
                    t.cancel()
                agent_bus.close()
                peer.close()
                broker.stop()


@unittest.skipUnless(HAVE_ZMQ, "pyzmq/protobuf not installed")
class KeyFileTests(unittest.TestCase):
    def test_world_readable_secret_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            _, secret = generate_keypair(d, "agent")
            self.assertEqual(stat.S_IMODE(os.stat(secret).st_mode), 0o600)
            os.chmod(secret, 0o644)
            with self.assertRaises(PermissionError):
                load_secret(str(secret))


if __name__ == "__main__":
    unittest.main()
