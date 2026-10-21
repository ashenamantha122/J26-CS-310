"""
Observe-mode score logging and the ROC sweep (proposal 3.5).
IT23184312 | Amantha M A
"""

from __future__ import annotations

import asyncio
import io
import json
import contextlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks"))

import roc_eval  # noqa: E402
from laax.backend import SimulatedBackend  # noqa: E402
from laax.bus import InProcBus  # noqa: E402
from laax.contracts import HostEvent  # noqa: E402
from services.c1_agent import C1Agent  # noqa: E402
from services.c1_scorelog import ScoreLog  # noqa: E402


def write_scores(path, rows):
    with open(path, "w") as fh:
        for ts, conf, learning in rows:
            fh.write(json.dumps({"ts": ts, "confidence": conf, "learning": learning,
                                 "scorer": os.path.basename(path)}) + "\n")


def write_labels(path, windows):
    with open(path, "w") as fh:
        fh.write("start,end,label,command\n")
        for s, e, l in windows:
            fh.write(f"{s},{e},{l},\"x\"\n")


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = roc_eval.main(argv)
    return rc, out.getvalue()


class RocTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.labels = os.path.join(self.d.name, "labels.csv")
        # three attack windows, benign activity in between
        write_labels(self.labels, [(100, 105, "T1"), (200, 205, "T2"), (300, 305, "T3")])

    def tearDown(self):
        self.d.cleanup()

    def benign(self, conf):
        return [(t, conf, False) for t in range(0, 400, 3)
                if not any(s - 2 <= t <= e + 2 for s, e in ((100, 105), (200, 205), (300, 305)))]

    def test_perfect_separation_gives_auc_1_and_a_threshold_between(self):
        path = os.path.join(self.d.name, "good.jsonl")
        write_scores(path, self.benign(1.0) + [(102, 9.0, False), (202, 8.0, False),
                                               (302, 10.0, False)])
        r = roc_eval.evaluate(path, roc_eval.load_labels(self.labels),
                              type("A", (), {"grace": 2.0, "bin": 10.0,
                                             "target_recall": 1.0, "theta": None})())
        self.assertAlmostEqual(r["auc"], 1.0)
        t, tpr, fpr = r["op"]
        self.assertEqual((tpr, fpr), (1.0, 0.0))
        self.assertGreater(t, 1.0)
        self.assertLessEqual(t, 8.0)

    def test_uninformative_scores_give_auc_half(self):
        path = os.path.join(self.d.name, "flat.jsonl")
        write_scores(path, self.benign(5.0) + [(102, 5.0, False), (202, 5.0, False),
                                               (302, 5.0, False)])
        rc, out = run(["--scores", path, "--labels", self.labels])
        self.assertEqual(rc, 0)
        self.assertIn("AUC                            0.500", out)

    def test_recall_first_picks_highest_threshold_meeting_target(self):
        path = os.path.join(self.d.name, "mixed.jsonl")
        # one attack scores low; one benign slice scores high
        write_scores(path, self.benign(1.0) + [(50, 9.5, False), (102, 9.0, False),
                                               (202, 8.0, False), (302, 3.0, False)])
        r = roc_eval.evaluate(path, roc_eval.load_labels(self.labels),
                              type("A", (), {"grace": 2.0, "bin": 10.0,
                                             "target_recall": 0.6, "theta": 7.0})())
        t, tpr, _fpr = r["op"]
        self.assertEqual(t, 8.0)                     # 2 of 3 windows, highest such t
        self.assertAlmostEqual(tpr, 2 / 3)
        self.assertEqual(r["missed"], ["T3"])
        self.assertAlmostEqual(r["at_theta"][1], 2 / 3)

    def test_learning_events_and_idle_time_do_not_count(self):
        path = os.path.join(self.d.name, "learn.jsonl")
        rows = [(t, 50.0, True) for t in range(0, 90)]          # learning: ignored
        rows += [(102, 9.0, False), (202, 9.0, False), (302, 9.0, False), (150, 1.0, False)]
        write_scores(path, rows)
        r = roc_eval.evaluate(path, roc_eval.load_labels(self.labels),
                              type("A", (), {"grace": 2.0, "bin": 10.0,
                                             "target_recall": 1.0, "theta": None})())
        self.assertEqual(r["benign_slices"], 1)       # only the slice with an event
        self.assertAlmostEqual(r["auc"], 1.0)


class ObserveModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_observe_scores_and_logs_but_never_switches(self):
        class Src:
            def __init__(self, evs):
                self.evs = evs

            def __aiter__(self):
                return self._g()

            async def _g(self):
                for e in self.evs:
                    yield e

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.jsonl")
            log = ScoreLog(path)
            evs = [HostEvent(host_id="h", pid=10 + i, ppid=1, comm=f"rare{i}",
                             exe=f"/tmp/rare{i}", kind="exec", hook="tracepoint")
                   for i in range(5)]
            agent = C1Agent(InProcBus(), Src(evs), host_id="h", backend=SimulatedBackend(),
                            observe=True, score_log=log)
            agent.scorer.freeze()                       # empty baseline: everything rare
            await agent.run()
            log.close()
            self.assertEqual(agent.ctl.switches, 0)
            self.assertTrue(any(s.confidence >= agent.ctl.theta_up for s in agent.procs.values()))
            with open(path) as fh:
                rows = [json.loads(l) for l in fh]
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["exe"], "/tmp/rare0")
            self.assertFalse(rows[0]["learning"])
            self.assertIn("explanation", rows[0])
            self.assertIn("ancestry", rows[0]["explanation"]["signals"])
            self.assertGreaterEqual(rows[0]["native_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
