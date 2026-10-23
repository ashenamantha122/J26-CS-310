"""PP1 Key Function 1: scoring-v2 signal and baseline guards."""

from __future__ import annotations

import unittest

from laax.bus import InProcBus
from laax.contracts import HostEvent
from services.c1_agent import C1Agent, RarityScorer, canon_comm


def event(**values) -> HostEvent:
    base = {"host_id": "h", "pid": 10, "ppid": 1, "comm": "curl", "kind": "exec"}
    base.update(values)
    return HostEvent(**base)


class ScoringV2Tests(unittest.TestCase):
    def test_two_step_ancestry_is_used_when_available(self):
        scorer = RarityScorer()
        ev = event(comm="sh")
        ev.comm_parent = "apache2"
        ev.comm_grandparent = "systemd"
        score = scorer.score_event(ev)

        self.assertGreater(score, 0)
        explanation = scorer.last_explanation["signals"]["ancestry"]
        self.assertEqual(explanation["why"], "two-step ancestry")
        self.assertEqual(explanation["key"], ["systemd", "apache2", "sh"])

    def test_one_step_ancestry_is_a_safe_fallback(self):
        scorer = RarityScorer()
        ev = event(comm="sh")
        ev.comm_parent = "apache2"
        scorer.score_event(ev)

        explanation = scorer.last_explanation["signals"]["ancestry"]
        self.assertEqual(explanation["why"], "one-step ancestry fallback")
        self.assertEqual(explanation["key"], ["apache2", "sh"])

    def test_egress_keys_on_process_zone_and_port(self):
        scorer = RarityScorer(zones={"iot": ["192.168.100.0/24"]})
        ev = event(kind="connect", daddr="192.168.100.42", dport=1883)
        scorer.score_event(ev)

        explanation = scorer.last_explanation["signals"]["egress"]
        self.assertEqual(explanation["key"], ["curl", "iot", 1883])

    def test_executable_path_is_an_independent_signal(self):
        scorer = RarityScorer()
        ev = event(exe="/tmp/.cache/dropper")
        ev.comm_parent = "apache2"
        scorer.score_event(ev)

        signals = scorer.last_explanation["signals"]
        self.assertIn("executable", signals)
        self.assertEqual(signals["executable"]["key"], ["/tmp/.cache/dropper"])
        self.assertGreaterEqual(scorer.last_explanation["native_score"], 0.0)
        self.assertLessEqual(scorer.last_explanation["native_score"], 1.0)

    def test_frozen_baseline_only_learns_calm_light_events(self):
        scorer = RarityScorer()
        known = event(comm="cron")
        known.comm_parent = "systemd"
        for _ in range(5):
            score = scorer.evaluate(known)
            self.assertTrue(scorer.learn(score, confidence_before=0.0, mode="light",
                                         confidence_limit=4.0))
        scorer.freeze()

        # Known, calm activity can adapt the baseline gradually, and its moving
        # baseline remains aligned with the profile as counts grow.
        before = scorer.baseline_rarity
        self.assertTrue(scorer.learn(scorer.evaluate(known), confidence_before=0.0,
                                     mode="light", confidence_limit=4.0))
        self.assertNotEqual(before, scorer.baseline_rarity)
        # A suspicious process or anything already in deep mode cannot poison it.
        score = scorer.evaluate(known)
        self.assertFalse(scorer.learn(score, confidence_before=5.0, mode="light",
                                      confidence_limit=4.0))
        self.assertFalse(scorer.learn(score, confidence_before=0.0, mode="deep",
                                      confidence_limit=4.0))

    def test_frozen_baseline_never_learns_a_known_name_at_a_new_path(self):
        """A copied binary must remain rare until an explicit retraining run."""
        scorer = RarityScorer()
        known = event(comm="ls", exe="/usr/bin/ls")
        known.comm_parent = "bash"
        known.comm_grandparent = "systemd"
        for _ in range(5):
            self.assertTrue(scorer.learn(
                scorer.evaluate(known), confidence_before=0.0, mode="light",
                confidence_limit=4.0))
        scorer.freeze()

        copied = event(comm="ls", exe="/tmp/ls")
        copied.comm_parent = "bash"
        copied.comm_grandparent = "systemd"
        score = scorer.evaluate(copied)

        self.assertFalse(scorer.learn(
            score, confidence_before=0.0, mode="light", confidence_limit=4.0))
        executable = next(signal for signal in score.signals
                          if signal.name == "executable")
        self.assertEqual(scorer.counts[executable.context].get(executable.key, 0), 0)

    def test_zone_longest_prefix_ipv6_and_unknown(self):
        scorer = RarityScorer(zones={
            "private": ["10.0.0.0/8"],
            "iot": ["10.4.0.0/16", "2001:db8:42::/48"],
        })
        self.assertEqual(scorer.destination_zone("10.4.8.9"), "iot")
        self.assertEqual(scorer.destination_zone("2001:db8:42::7"), "iot")
        self.assertEqual(scorer.destination_zone("not-an-address"), "unknown")
        self.assertEqual(scorer.destination_zone(None), "unknown")


class AgentAncestryLookupTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_keeps_kernel_grandparent_when_local_lookup_is_unknown(self):
        class Source:
            def __aiter__(self):
                return self._events()

            async def _events(self):
                yield shell

        shell = event(pid=11, ppid=10, comm="ls", exe="/usr/bin/ls")
        shell.comm_parent = "bash"
        shell.comm_grandparent = "kernel-systemd"
        agent = C1Agent(InProcBus(), Source(), host_id="h")
        # The agent saw the parent but has no name for its grandparent. This is
        # the common long-lived-service case that must not erase kernel gpcomm.
        agent.parent_by_pid[10] = 1
        await agent.run()

        ancestry = next(signal for signal in agent.scorer.evaluate(shell).signals
                        if signal.name == "ancestry")
        self.assertEqual(shell.comm_grandparent, "kernel-systemd")
        self.assertEqual(ancestry.context, "ancestry_2")
        self.assertEqual(ancestry.key, ("kernel-systemd", "bash", "ls"))

    async def test_agent_derives_grandparent_from_its_own_process_table(self):
        class Source:
            def __init__(self, events):
                self.events = events

            def __aiter__(self):
                return self._events()

            async def _events(self):
                for item in self.events:
                    yield item

        systemd = event(pid=1, ppid=0, comm="systemd")
        apache = event(pid=10, ppid=1, comm="apache2")
        apache.comm_parent = "systemd"
        shell = event(pid=11, ppid=10, comm="sh")
        shell.comm_parent = "apache2"
        agent = C1Agent(InProcBus(), Source([systemd, apache, shell]), host_id="h")
        await agent.run()

        # Normal agent operation keeps the explanation off the hot path.
        self.assertEqual(agent.scorer.last_explanation, {})
        ancestry = next(signal for signal in agent.scorer.evaluate(shell).signals
                        if signal.name == "ancestry")
        self.assertEqual(ancestry.context, "ancestry_2")
        self.assertEqual(ancestry.key, ("systemd", "apache2", "sh"))


class CommCanonicalisationTests(unittest.TestCase):
    """Volatile per-thread serials must collapse to one stable rarity key.

    Regression guard for the 22 Oct 2026 ROC finding: Firefox resolver threads
    (``DNS Res~er #<n>``) each produced a unique key and dominated both the
    attack windows and the benign false-positive wall.
    """

    def test_canon_strips_thread_serial(self):
        self.assertEqual(canon_comm("DNS Res~er #648"), "DNS Res~er")
        self.assertEqual(canon_comm("DNS Res~er #646"), "DNS Res~er")

    def test_canon_folds_kworker_pool(self):
        self.assertEqual(canon_comm("kworker/u16:3-events"), "kworker")

    def test_canon_preserves_distinct_programs(self):
        # Conservative: version suffixes and ordinary names are untouched.
        self.assertEqual(canon_comm("python3.11"), "python3.11")
        self.assertEqual(canon_comm("curl"), "curl")
        self.assertIsNone(canon_comm(None))

    def test_resolver_threads_share_one_egress_key(self):
        scorer = RarityScorer()
        first = event(kind="connect", comm="DNS Res~er #648",
                      daddr="8.8.8.8", dport=53)
        second = event(kind="connect", comm="DNS Res~er #991",
                       daddr="8.8.8.8", dport=53)
        key1 = scorer.evaluate(first).signals[0].key
        key2 = scorer.evaluate(second).signals[0].key
        self.assertEqual(key1, key2)

    def test_learning_one_resolver_thread_tames_the_next(self):
        # After the pool is seen once, a sibling thread is no longer "novel".
        scorer = RarityScorer()
        seen = scorer.evaluate(event(kind="connect", comm="DNS Res~er #648",
                                     daddr="8.8.8.8", dport=53)).signals[0]
        for _ in range(50):
            scorer.observe(seen.context, seen.key)
        novel_raw = scorer.rarity("egress", ("DNS Res~er #777", "public", 53))
        novel_canon = scorer.evaluate(
            event(kind="connect", comm="DNS Res~er #777",
                  daddr="8.8.8.8", dport=53)).signals[0].rarity
        self.assertLess(novel_canon, novel_raw)


if __name__ == "__main__":
    unittest.main()
