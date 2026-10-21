"""
Tests for the C1 -> C3 telemetry export (services/c1_telemetry.py).

    python -m unittest discover -s tests -v

Standard library only, so they run on Windows and inside WSL without installs.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import os
import tempfile
import unittest
from datetime import datetime

from laax.bus import InProcBus
from laax.contracts import ExternalAlert, HostEvent, Topic
from scenario.lateral_movement import ScriptedKernelSource
from services.c1_agent import C1Agent
from services.c1_telemetry import (
    COLUMNS,
    NULL,
    TIME_FORMAT,
    TelemetryCsvExporter,
    check_rows,
    normalise_confidence,
    to_row,
)

# The header of C1_Output_for_C3_V6_v2_2019.csv, the reference file agreed
# with Component 3. Changing COLUMNS without changing this is a contract break.
REFERENCE_HEADER = (
    "timestamp,host_id,host_ip,process_id,parent_process_id,process_name,"
    "event_type,trace_mode,trace_source,src_ip,dst_ip,src_port,dst_port,"
    "protocol,syscall_name,syscall_result,process_state,memory_alloc_bytes,"
    "file_operation,threat_confidence_score,threat_state,cpu_usage,"
    "disk_read_bytes,disk_write_bytes,label,type"
)
HOST_IP = "10.0.0.12"
THETA_UP, THETA_DOWN = 7.0, 4.0


def row(**kw) -> dict:
    ev = HostEvent(host_id="h1", pid=4471, ppid=1400, comm="bash", **kw)
    return to_row(ev, HOST_IP, THETA_UP, THETA_DOWN)


class SchemaTests(unittest.TestCase):
    def test_columns_match_reference_header(self):
        self.assertEqual(",".join(COLUMNS), REFERENCE_HEADER)

    def test_light_exec_row(self):
        r = row(kind="exec", depth="light", hook="tracepoint")
        self.assertEqual(r["event_type"], "process_start")
        self.assertEqual(r["trace_source"], "tracepoint")
        self.assertEqual(r["syscall_name"], "execve")
        self.assertEqual(r["syscall_result"], "success")
        self.assertEqual((r["src_ip"], r["dst_ip"], r["protocol"]), (NULL, NULL, NULL))
        self.assertEqual((r["src_port"], r["dst_port"]), ("", ""))
        self.assertEqual(r["host_ip"], HOST_IP)

    def test_connect_falls_back_to_host_ip_and_leaves_unknown_port_empty(self):
        r = row(kind="connect", hook="tracepoint", daddr="66.42.13.90", dport=4444,
                proto="tcp")
        self.assertEqual(r["event_type"], "network_connect")
        self.assertEqual((r["src_ip"], r["dst_ip"]), (HOST_IP, "66.42.13.90"))
        self.assertEqual((r["src_port"], r["dst_port"]), ("", 4444))
        self.assertEqual(r["protocol"], "TCP")

    def test_unknown_protocol_is_not_guessed(self):
        r = row(kind="connect", hook="tracepoint", daddr="::1", dport=53)
        self.assertEqual(r["protocol"], NULL)
        r = row(kind="connect", hook="tracepoint", daddr="::1", dport=53, proto="udp")
        self.assertEqual(r["protocol"], "UDP")
        self.assertEqual(r["syscall_result"], NULL)   # entry tracepoint: result unknown

    def test_accept_puts_remote_peer_in_src(self):
        r = row(kind="accept", hook="tracepoint", saddr="66.42.13.90",
                daddr=HOST_IP, sport=51000, dport=22, syscall="accept4")
        self.assertEqual((r["src_ip"], r["dst_ip"]), ("66.42.13.90", HOST_IP))
        self.assertEqual((r["src_port"], r["dst_port"]), (51000, 22))
        self.assertEqual(r["syscall_name"], "accept")

    def test_negative_ret_is_failure(self):
        self.assertEqual(row(kind="exec", hook="kprobe", ret=-13)["syscall_result"], "failure")

    def test_ground_truth_only_when_supplied(self):
        self.assertEqual((row(kind="exec")["label"], row(kind="exec")["type"]), (NULL, NULL))
        r = row(kind="exec", label="normal")
        self.assertEqual((r["label"], r["type"]), (0, "normal"))
        r = row(kind="exec", label="lateral_movement")
        self.assertEqual((r["label"], r["type"]), (1, "lateral_movement"))

    def test_timestamp_format(self):
        r = row(kind="exec", ts=datetime(2026, 10, 15, 9, 5, 7).timestamp())
        self.assertEqual(r["timestamp"], "15/10/2026 09:05:07")
        datetime.strptime(r["timestamp"], TIME_FORMAT)

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            row(kind="sample")


class ScoreTests(unittest.TestCase):
    def test_bounds_and_anchor(self):
        self.assertEqual(normalise_confidence(0.0, THETA_UP), 0.0)
        self.assertEqual(normalise_confidence(-3.0, THETA_UP), 0.0)
        self.assertAlmostEqual(normalise_confidence(THETA_UP, THETA_UP), 0.6, places=9)
        self.assertLess(normalise_confidence(1e6, THETA_UP), 1.0 + 1e-12)

    def test_monotonic_so_thresholds_survive(self):
        xs = [0.1 * i for i in range(300)]
        ys = [normalise_confidence(x, THETA_UP) for x in xs]
        self.assertEqual(ys, sorted(ys))

    def test_threat_state_follows_lower_threshold(self):
        self.assertEqual(row(kind="exec", confidence=THETA_DOWN)["threat_state"], "normal")
        self.assertEqual(row(kind="exec", confidence=THETA_DOWN + 0.01)["threat_state"],
                         "suspicious")


class CheckerTests(unittest.TestCase):
    def good(self, **over):
        r = {c: NULL for c in COLUMNS}
        r.update(timestamp="15/10/2026 09:05:07", trace_mode="light",
                 trace_source="tracepoint", event_type="process_start",
                 threat_confidence_score="0.2")
        r.update(over)
        return r

    def test_clean_rows_pass(self):
        self.assertEqual(check_rows(COLUMNS, [self.good()]), {})

    def test_kprobe_in_light_mode_is_flagged(self):
        bad = check_rows(COLUMNS, [self.good(trace_source="kprobe")])
        self.assertIn("light mode row from hook 'kprobe'", bad)

    def test_xdp_is_never_valid(self):
        bad = check_rows(COLUMNS, [self.good(trace_mode="deep", trace_source="xdp")])
        self.assertIn("deep mode row from hook 'xdp'", bad)

    def test_wrong_header_is_flagged(self):
        self.assertIn("header differs from the C3 schema",
                      check_rows(COLUMNS[::-1], []))


class AgentExportTests(unittest.TestCase):
    """The agent's real controller drives trace_mode; the export must reflect it."""

    def test_agent_output_conforms_and_shows_the_mode_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c1.csv")
            with contextlib.redirect_stdout(io.StringIO()):
                rows = asyncio.run(self._run_agent(path))
            with open(path, newline="", encoding="utf-8") as fh:
                header = fh.readline().strip()

        self.assertEqual(header, REFERENCE_HEADER)
        self.assertEqual(check_rows(COLUMNS, rows), {})

        by_pid = {}
        for r in rows:
            by_pid.setdefault(r["process_id"], []).append(r)
        # Benign baseline: light only, no payload hooks.
        self.assertTrue(all(r["trace_mode"] == "light" for r in by_pid["1100"]))
        # sshd promoted by C3's alert; its child is born deep and its payload
        # comes from the deep kprobe.
        child = by_pid["4471"][0]
        self.assertEqual((child["trace_mode"], child["trace_source"]), ("deep", "kprobe"))
        self.assertEqual(child["threat_state"], "suspicious")
        self.assertGreaterEqual(float(child["threat_confidence_score"]), 0.6 * 0.7)
        # An unrelated process stays light while sshd's subtree is deep.
        late = by_pid["1200"][-1]
        self.assertEqual((late["trace_mode"], late["trace_source"]), ("light", "tracepoint"))

    async def _run_agent(self, path):
        bus = InProcBus()
        src = ScriptedKernelSource()
        agent = C1Agent(bus, src, host_id="h1")
        exporter = TelemetryCsvExporter(bus, path, {"h1": HOST_IP},
                                        agent.ctl.theta_up, agent.ctl.theta_down)
        tasks = [asyncio.create_task(bus.run()), asyncio.create_task(agent.run())]

        async def settle():
            for _ in range(3):
                while not src.q.empty():
                    await asyncio.sleep(0)
                await asyncio.sleep(0)
                await bus.drain()

        for _ in range(20):    # benign window
            await src.feed(HostEvent(pid=1100, ppid=1, comm="nginx", kind="connect",
                                     daddr="10.0.0.20", dport=5432, label="normal"))
            await src.feed(HostEvent(pid=1400, ppid=1, comm="sshd", kind="accept",
                                     saddr="10.0.0.30", daddr=HOST_IP, dport=22,
                                     label="normal"))
        await settle()
        agent.scorer.freeze()

        await bus.publish(Topic.EXTERNAL_ALERT, ExternalAlert(
            host_id="h1", pid=1400, boost=10.0, reason="test"))
        await settle()
        await src.feed(HostEvent(pid=4471, ppid=1400, comm="bash", kind="exec",
                                 argv=["/bin/bash", "-i"], label="lateral_movement"))
        await src.feed(HostEvent(pid=1200, ppid=1, comm="postgres", kind="connect",
                                 daddr="10.0.0.21", dport=5432, label="normal"))
        await settle()

        for t in tasks:
            t.cancel()
        exporter.close()
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))


if __name__ == "__main__":
    unittest.main()
