"""
Unit and integration tests for Component 1 Agent lifecycle and standalone runner.
Student: Amantha M.A (IT23184312) | Research Project: J26-CS-310

Tests:
  1. Standalone agent initialization & configuration
  2. ModeController hysteresis (theta_up, theta_down, min_dwell_s)
  3. CUSUM confidence accumulation vs decay
  4. ExternalAlert E(t) feedback boost & escalation
  5. EnforceHost LSM syscall denial arming and timed expiry
  6. Autonomous SyntheticLiveSource injection & processing
"""

from __future__ import annotations

import asyncio
import unittest

from laax.backend import SimulatedBackend
from laax.bus import InProcBus
from laax.contracts import EnforceHost, ExternalAlert, HostEvent, Topic
from laax_agent import SyntheticLiveSource
from services.c1_agent import C1Agent, ModeController, ProcState, RarityScorer


class AgentLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bus = InProcBus()
        self.source = SyntheticLiveSource(host_id="test-host")
        self.ctl = ModeController(theta_up=7.0, theta_down=4.0, min_dwell_s=0.2)
        self.agent = C1Agent(
            bus=self.bus,
            source=self.source,
            host_id="test-host",
            controller=self.ctl,
            backend=SimulatedBackend(attach_ms=10.0, detach_ms=150.0, map_ms=0.01),
        )

    async def test_rarity_learning_and_freeze(self):
        scorer = self.agent.scorer
        self.assertTrue(scorer.learning)

        # Observe benign pairs
        ev1 = HostEvent(kind="exec", comm="bash", host_id="test-host")
        setattr(ev1, "comm_parent", "systemd")
        r1 = scorer.score_event(ev1)
        self.assertGreater(r1, 0.0)

        scorer.freeze()
        self.assertFalse(scorer.learning)
        self.assertGreater(scorer.baseline_rarity, 0.0)

    async def test_mode_controller_hysteresis(self):
        st = ProcState(pid=1001, comm="testproc", mode="light")

        # Below threshold -> no switch
        st.confidence = 5.0
        self.assertIsNone(self.ctl.decide(st))

        # Reaches theta_up (7.0) -> switches to deep
        st.confidence = 7.5
        decision = self.ctl.decide(st)
        self.assertEqual(decision, ("deep", "rarity"))

        # Transition to deep
        self.ctl.record(st, "deep", latency_ms=10.0, global_transition=True)
        self.assertEqual(st.mode, "deep")

        # Immediate score drop below theta_down -> blocked by min_dwell_s floor!
        st.confidence = 2.0
        self.assertIsNone(self.ctl.decide(st), "Must not flap before min_dwell_s expires")

        # Wait out dwell time
        await asyncio.sleep(0.25)
        decision = self.ctl.decide(st)
        self.assertEqual(decision, ("light", "dwell_expiry"), "Should de-escalate after dwell time")

    async def test_external_alert_boost(self):
        st = ProcState(pid=2002, comm="sshd", mode="light", confidence=2.0)
        self.agent.procs[2002] = st

        alert = ExternalAlert(
            host_id="test-host",
            pid=2002,
            boost=6.0,
            reason="C3 GNN correlation detected",
        )
        await self.agent.on_external_alert(Topic.EXTERNAL_ALERT, alert)

        self.assertEqual(st.confidence, 8.0)
        self.assertEqual(st.mode, "deep", "Process should escalate to deep mode after E(t) boost")
        self.assertIn(2002, self.agent.deep_pids)

    async def test_lsm_enforce_and_timed_expiry(self):
        # C4 may only target a pid C1 has reported.
        self.agent.procs[3003] = ProcState(pid=3003, comm="bash")
        enforce = EnforceHost(
            policy_id="pol-test-1",
            host_id="test-host",
            pid=3003,
            syscall="execve",
            lease_seconds=0.15,
        )
        await self.agent.on_enforce(Topic.ENFORCE_HOST, enforce)

        # Check denial armed
        self.assertIn("execve", self.agent.denied[3003])

        # Await lease expiration
        await asyncio.sleep(0.25)
        self.assertNotIn("execve", self.agent.denied[3003], "Denial must be released after lease expiry")

    async def test_process_tree_inheritance(self):
        # Parent process in deep mode
        parent_st = ProcState(pid=4000, comm="bash", mode="deep", confidence=9.0)
        self.agent.procs[4000] = parent_st
        self.agent.deep_pids.add(4000)

        bus_task = asyncio.create_task(self.bus.run())
        agent_task = asyncio.create_task(self.agent.run())

        # Inject child process spawned by deep parent
        self.source.inject("nc", kind="exec", ppid=4000, pid=4001, argv=["nc", "-l", "9999"], pcomm="bash")
        await asyncio.sleep(0.1)

        child_st = self.agent.procs.get(4001)
        self.assertIsNotNone(child_st)
        self.assertEqual(child_st.mode, "deep", "Child spawned by deep parent must inherit deep mode")
        self.assertGreater(child_st.confidence, 0.0)

        bus_task.cancel()
        agent_task.cancel()


if __name__ == "__main__":
    unittest.main()
