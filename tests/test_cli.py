"""
laax-agent commands: version, check, status, keygen, verify-audit, and the
telemetry-only degrade when the BPF LSM is missing.
IT23184312 | Amantha M A
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest

import laax_agent
from laax.backend import SimulatedBackend
from laax.bus import InProcBus
from laax.contracts import EnforceHost, Topic
from services import c1_audit
from services.c1_agent import C1Agent, ProcState


def run(argv) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = laax_agent.main(argv)
    return rc, out.getvalue()


class CommandTests(unittest.TestCase):
    def test_version(self):
        rc, out = run(["version"])
        self.assertEqual(rc, 0)
        self.assertIn(laax_agent.__version__, out)

    def test_status_without_agent_is_rc3(self):
        rc, out = run(["status", "--status-file", "/nonexistent/laax.json"])
        self.assertEqual(rc, 3)
        self.assertIn("Is the agent running", out)

    def test_status_reports_running_for_live_pid_and_stale_otherwise(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "status.json")
            st = {"state": "running", "pid": os.getpid(), "updated": time.time(),
                  "version": "x", "stats": {"events_emitted": 5}}
            with open(path, "w") as fh:
                json.dump(st, fh)
            rc, out = run(["status", "--status-file", path])
            self.assertEqual(rc, 0)
            self.assertIn("running", out.splitlines()[0])
            st["updated"] = time.time() - 120          # heartbeat stopped
            with open(path, "w") as fh:
                json.dump(st, fh)
            rc, out = run(["status", "--status-file", path])
            self.assertEqual(rc, 3)
            self.assertIn("NOT RUNNING", out)

    @unittest.skipUnless(os.name == "posix", "key file modes")
    def test_keygen_server_then_client_and_no_overwrite(self):
        try:
            import zmq  # noqa: F401
        except ImportError:
            self.skipTest("pyzmq not installed")
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run(["keygen", d, "--server"])[0], 0)
            self.assertTrue(os.path.isdir(os.path.join(d, "clients")))
            self.assertEqual(run(["keygen", d, "--name", "agent"])[0], 0)
            mode = os.stat(os.path.join(d, "agent.key_secret")).st_mode & 0o777
            self.assertEqual(mode, 0o600)
            self.assertEqual(run(["keygen", d, "--server"])[0], 1)   # refuses to overwrite

    def test_verify_audit(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.jsonl")
            log = c1_audit.AuditLog(path)
            log.write("agent_start")
            log.close()
            self.assertEqual(run(["verify-audit", path])[0], 0)
            with open(path, "a") as fh:
                fh.write('{"event":"forged","chain":"00"}\n')
            self.assertEqual(run(["verify-audit", path])[0], 1)

    def test_bad_config_is_rc2(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.conf")
            with open(path, "w") as fh:
                json.dump({"bus": {"type": "zmq", "curve_secret_key": "/x"}}, fh)
            rc, out = run(["--config", path])
            self.assertEqual(rc, 2)
            self.assertIn("CURVE needs both", out)


class DegradedEnforcementTests(unittest.IsolatedAsyncioTestCase):
    async def test_without_lsm_requests_are_refused_and_recorded(self):
        class NoLsm(SimulatedBackend):
            enforcement_available = False

        agent = C1Agent(InProcBus(), source=None, host_id="h", backend=NoLsm())
        agent.procs[10] = ProcState(pid=10, comm="sshd")
        await agent.on_enforce(Topic.ENFORCE_HOST, EnforceHost(host_id="h", pid=10,
                                                               policy_id="p"))
        self.assertFalse(agent.denied.get(10))
        rejected = [r for r in agent.audit.records if r["event"] == "enforce_rejected"]
        self.assertIn("not available", rejected[0]["reason"])


if __name__ == "__main__":
    unittest.main()
