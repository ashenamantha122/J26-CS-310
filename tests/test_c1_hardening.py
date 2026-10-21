"""
Tests for the 21 Oct 2026 hardening of Component 1.
IT23184312 | Amantha M A

Each test pins one defect found in the audit, so it cannot come back:
  - enforcement: unsupported syscalls and unknown pids refused, leases per
    policy, inherited denials released with the policy, rollback
  - process lifecycle: fork inheritance, exit cleanup
  - deep kprobe events do not feed the score
  - transitions serialised (promotion during a detach keeps probes attached)
  - audit log hash chain, config file, bus authentication, kernel event parsing
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest

from laax.backend import SimulatedBackend
from laax.bus import Broker, BusAuthError, InProcBus, TcpBus
from laax.contracts import EnforceHost, HostEvent, ModeChange, Rollback, Topic
from services import c1_audit
from services.c1_agent import C1Agent, ModeController, ProcState
from services.c1_telemetry import to_row
from services.ebpf_source import EbpfKernelSource

import laax_agent


class RecordingBackend(SimulatedBackend):
    """Cost model that also records every kernel operation it was asked for."""

    def __init__(self, detach_delay: float = 0.0) -> None:
        super().__init__(attach_ms=1.0, detach_ms=2.0, map_ms=0.01)
        self.calls: list[tuple] = []
        self.detach_delay = detach_delay

    async def attach_deep(self) -> float:
        self.calls.append(("attach",))
        return await super().attach_deep()

    async def detach_deep(self) -> float:
        self.calls.append(("detach",))
        await asyncio.sleep(self.detach_delay)
        return await super().detach_deep()

    async def gate_add(self, pid: int) -> float:
        self.calls.append(("gate_add", pid))
        return self.map_ms

    async def gate_del(self, pid: int) -> float:
        self.calls.append(("gate_del", pid))
        return self.map_ms

    async def deny_add(self, pid: int, syscall: str) -> float:
        self.calls.append(("deny_add", pid))
        return await super().deny_add(pid, syscall)

    async def deny_del(self, pid: int) -> float:
        self.calls.append(("deny_del", pid))
        return self.map_ms


def make_agent(backend=None) -> C1Agent:
    return C1Agent(InProcBus(), source=None, host_id="h",
                   controller=ModeController(min_dwell_s=0.0),
                   backend=backend or RecordingBackend())


def enforce(pid: int, policy: str = "p1", lease: float = 60.0,
            syscall: str = "execve") -> EnforceHost:
    return EnforceHost(policy_id=policy, host_id="h", pid=pid,
                       syscall=syscall, lease_seconds=lease)


class EnforcementTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.be = RecordingBackend()
        self.agent = make_agent(self.be)
        self.agent.procs[100] = ProcState(pid=100, comm="sshd")

    def events(self, name):
        return [r for r in self.agent.audit.records if r["event"] == name]

    async def test_unsupported_syscall_is_refused_not_armed_as_execve(self):
        await self.agent.on_enforce(Topic.ENFORCE_HOST, enforce(100, syscall="connect"))
        self.assertNotIn(("deny_add", 100), self.be.calls)
        self.assertFalse(self.agent.denied.get(100))
        self.assertEqual(len(self.events("enforce_rejected")), 1)

    async def test_pid_never_reported_by_c1_is_refused(self):
        await self.agent.on_enforce(Topic.ENFORCE_HOST, enforce(4242))
        self.assertEqual(self.be.calls, [])
        self.assertIn("never reported", self.events("enforce_rejected")[0]["reason"])

    async def test_overlapping_leases_first_expiry_does_not_lift_second(self):
        await self.agent.on_enforce(Topic.ENFORCE_HOST, enforce(100, "short", 0.05))
        await self.agent.on_enforce(Topic.ENFORCE_HOST, enforce(100, "long", 0.3))
        await asyncio.sleep(0.12)
        self.assertNotIn(("deny_del", 100), self.be.calls)
        self.assertIn("execve", self.agent.denied[100])
        await asyncio.sleep(0.3)
        self.assertIn(("deny_del", 100), self.be.calls)
        self.assertNotIn(100, self.agent.denied)

    async def test_inherited_denial_is_released_with_the_policy(self):
        await self.agent.on_enforce(Topic.ENFORCE_HOST, enforce(100, "p1", 0.1))
        fork = HostEvent(host_id="h", pid=101, ppid=100, comm="sshd", kind="fork")
        fork.inherit_flags = 0x2
        await self.agent._on_fork(fork)
        self.assertIn(101, self.agent.leases["p1"]["pids"])
        await asyncio.sleep(0.2)
        self.assertIn(("deny_del", 100), self.be.calls)
        self.assertIn(("deny_del", 101), self.be.calls)
        released = self.events("enforce_released")[0]
        self.assertEqual(released["pids"], [100, 101])

    async def test_rollback_releases_early(self):
        await self.agent.on_enforce(Topic.ENFORCE_HOST, enforce(100, "p1", 60))
        await self.agent.on_rollback(Topic.ROLLBACK,
                                     Rollback(policy_id="p1", reason="disproved"))
        self.assertIn(("deny_del", 100), self.be.calls)
        self.assertEqual(self.events("enforce_released")[0]["reason"], "disproved")
        self.assertNotIn("p1", self.agent.leases)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_fork_inherits_deep_without_a_second_attach(self):
        be = RecordingBackend()
        agent = make_agent(be)
        parent = ProcState(pid=10, comm="bash", confidence=9.0)
        agent.procs[10] = parent
        await agent._switch(parent, "deep", "rarity")
        fork = HostEvent(host_id="h", pid=11, ppid=10, comm="bash", kind="fork")
        fork.inherit_flags = 0x1
        await agent._on_fork(fork)
        self.assertEqual(agent.procs[11].mode, "deep")
        self.assertEqual(be.calls.count(("attach",)), 1)
        self.assertEqual(agent.deep_pids, {10, 11})

    async def test_exit_while_deep_keeps_probes_until_dwell_then_forgets(self):
        be = RecordingBackend()
        agent = make_agent(be)
        st = ProcState(pid=20, comm="nc", confidence=9.0)
        agent.procs[20] = st
        await agent._switch(st, "deep", "rarity")
        await agent._on_exit(HostEvent(host_id="h", pid=20, kind="exit"))
        # Not detached on exit: a burst of short-lived processes shares one
        # attach instead of paying a detach per process.
        self.assertNotIn(("detach",), be.calls)
        self.assertTrue(agent.procs[20].exited)
        decay = asyncio.create_task(agent.decay_loop(tick=0.01))
        await asyncio.sleep(0.1)
        decay.cancel()
        self.assertIn(("detach",), be.calls)
        self.assertNotIn(("gate_del", 20), be.calls)   # kernel already removed it
        self.assertNotIn(20, agent.procs)
        self.assertEqual(agent.deep_pids, set())
        self.assertFalse(agent.probes_attached)

    async def test_burst_of_short_deep_processes_shares_one_attach(self):
        be = RecordingBackend()
        agent = make_agent(be)
        agent.ctl.min_dwell_s = 0.2
        for pid in (40, 41, 42):
            st = ProcState(pid=pid, comm="x", confidence=9.0)
            agent.procs[pid] = st
            await agent._switch(st, "deep", "rarity")
            await agent._on_exit(HostEvent(host_id="h", pid=pid, kind="exit"))
        self.assertEqual(be.calls.count(("attach",)), 1)
        self.assertNotIn(("detach",), be.calls)

    async def test_reused_pid_closes_old_deep_window_without_touching_gate(self):
        be = RecordingBackend()
        agent = make_agent(be)
        old = ProcState(pid=60, comm="old", confidence=9.0)
        agent.procs[60] = old
        await agent._switch(old, "deep", "rarity")
        await agent._on_exit(HostEvent(host_id="h", pid=60, kind="exit"))
        await agent._retire_if_reused(60)
        self.assertNotIn(60, agent.procs)
        self.assertNotIn(("gate_del", 60), be.calls)
        self.assertIn(("detach",), be.calls)

    async def test_recycled_pid_starts_clean(self):
        agent = make_agent()
        agent.procs[30] = ProcState(pid=30, comm="old", confidence=50.0)
        await agent._on_exit(HostEvent(host_id="h", pid=30, kind="exit"))
        self.assertNotIn(30, agent.procs)

    async def test_promotion_during_detach_keeps_probes_attached(self):
        be = RecordingBackend(detach_delay=0.1)
        agent = make_agent(be)
        a = ProcState(pid=1, comm="a", confidence=9.0)
        b = ProcState(pid=2, comm="b", confidence=9.0)
        agent.procs.update({1: a, 2: b})
        await agent._switch(a, "deep", "rarity")
        leaving = asyncio.create_task(agent._switch(a, "light", "decay"))
        await asyncio.sleep(0.02)                 # detach is now in flight
        await agent._switch(b, "deep", "rarity")
        await leaving
        self.assertEqual(agent.deep_pids, {2})
        self.assertTrue(agent.probes_attached)
        self.assertEqual(be.calls[-1], ("attach",))


class ScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_kprobe_events_do_not_move_confidence(self):
        class Src:
            def __init__(self, evs):
                self.evs = evs

            def __aiter__(self):
                return self._gen()

            async def _gen(self):
                for e in self.evs:
                    yield e

        deep = HostEvent(host_id="h", pid=50, ppid=0, comm="python3", kind="exec",
                         hook="kprobe", depth="deep", argv=["/bin/true"])
        agent = C1Agent(InProcBus(), Src([deep]), host_id="h",
                        backend=RecordingBackend())
        agent.scorer.freeze()
        await agent.run()
        self.assertEqual(agent.procs[50].confidence, 0.0)


class AuditTests(unittest.TestCase):
    def test_chain_verifies_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            log = c1_audit.AuditLog(path)
            log.write("mode_change", pid=1)
            log.close()
            log = c1_audit.AuditLog(path)          # agent restart
            log.write("enforce_armed", pid=1)
            log.close()
            ok, n, _ = c1_audit.verify(path)
            self.assertTrue(ok)
            self.assertEqual(n, 2)

    def test_edit_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            log = c1_audit.AuditLog(path)
            for i in range(3):
                log.write("mode_change", pid=i)
            log.close()
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
            lines[1] = lines[1].replace('"pid":1', '"pid":9')
            with open(path, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
            ok, _, msg = c1_audit.verify(path)
            self.assertFalse(ok)
            self.assertIn("line 2", msg)

    def test_deleting_a_line_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            log = c1_audit.AuditLog(path)
            for i in range(3):
                log.write("mode_change", pid=i)
            log.close()
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
            with open(path, "w", encoding="utf-8") as fh:
                fh.writelines([lines[0], lines[2]])
            self.assertFalse(c1_audit.verify(path)[0])


class ConfigTests(unittest.TestCase):
    def write(self, d, obj):
        path = os.path.join(d, "agent.conf")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
        return path

    def test_shipped_config_loads_and_validates(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg = laax_agent.load_config(os.path.join(here, "deploy", "laax-agent.conf"))
        parser = laax_agent.build_parser()
        parser.set_defaults(**cfg)
        args = parser.parse_args([])
        laax_agent.validate(args)
        self.assertEqual(args.theta_up, 7.0)
        self.assertEqual(args.broker, "127.0.0.1:5599")
        self.assertEqual(args.audit, "/var/log/laax/audit.jsonl")

    def test_cli_overrides_config(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, {"thresholds": {"theta_up": 9.0}})
            parser = laax_agent.build_parser()
            parser.set_defaults(**laax_agent.load_config(path))
            self.assertEqual(parser.parse_args([]).theta_up, 9.0)
            self.assertEqual(parser.parse_args(["--theta-up", "8"]).theta_up, 8.0)

    def test_unknown_key_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, {"thresholds": {"theta_upp": 9.0}})
            with self.assertRaises(laax_agent.ConfigError):
                laax_agent.load_config(path)

    def test_bool_is_not_a_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, {"thresholds": {"theta_up": True}})
            with self.assertRaises(laax_agent.ConfigError):
                laax_agent.load_config(path)

    def test_inverted_hysteresis_is_rejected(self):
        args = laax_agent.build_parser().parse_args(["--theta-up", "3", "--theta-down", "4"])
        with self.assertRaises(laax_agent.ConfigError):
            laax_agent.validate(args)


class BusAuthTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.broker = Broker(port=0, token="s3cret")
        self.server = await asyncio.start_server(self.broker._client, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()

    async def test_right_token_delivers(self):
        got = asyncio.Event()

        async def handler(_t, _m):
            got.set()

        sub = TcpBus(port=self.port, token="s3cret")
        sub.subscribe(Topic.HOST_MODE, handler)
        task = asyncio.create_task(sub.run())
        await asyncio.sleep(0.1)
        pub = TcpBus(port=self.port, token="s3cret")
        await pub.publish(Topic.HOST_MODE, ModeChange())
        await asyncio.wait_for(got.wait(), 2.0)
        task.cancel()

    async def test_wrong_token_is_disconnected(self):
        bad = TcpBus(port=self.port, token="guess")
        bad.subscribe(Topic.ENFORCE_HOST, None)
        with self.assertRaises(BusAuthError):
            await asyncio.wait_for(bad.run(), 2.0)
        self.assertEqual(self.broker.rejected, 1)


class KernelParsingTests(unittest.IsolatedAsyncioTestCase):
    async def parse(self, d):
        class FakeLoader:
            events = asyncio.Queue()
        FakeLoader.events.put_nowait(d)
        return await EbpfKernelSource(FakeLoader(), host_id="h").__anext__()

    def base(self, **kw):
        d = {"pid": 7, "ppid": 1, "comm": "nc", "pcomm": "bash", "kind": "exec",
             "depth": "deep", "daddr": "", "saddr": "", "dport": 0, "sport": 0,
             "argv": "", "flags": 0, "ret": 0}
        d.update(kw)
        return d

    async def test_argv_is_split_on_the_separator(self):
        ev = await self.parse(self.base(argv="nc\x1f-e\x1f/bin/sh -i"))
        self.assertEqual(ev.argv, ["nc", "-e", "/bin/sh -i"])
        self.assertEqual(ev.hook, "kprobe")

    async def test_fork_is_lifecycle_not_a_kprobe(self):
        ev = await self.parse(self.base(kind="fork", flags=3))
        self.assertEqual(ev.hook, "tracepoint")
        self.assertEqual(ev.inherit_flags, 3)

    async def test_file_event_row(self):
        ev = await self.parse(self.base(kind="file", argv="shadow", ret=0))
        ev.ret = 2                                 # O_RDWR
        row = to_row(ev, "10.0.0.12", 7.0, 4.0)
        self.assertEqual(row["event_type"], "file_event")
        self.assertEqual(row["file_operation"], "open_readwrite")


if __name__ == "__main__":
    unittest.main()
