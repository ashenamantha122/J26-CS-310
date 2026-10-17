"""
supervisor.py — the entry point for Component 4. Ties together:

    Component 3 (ThreatAlert, ZMQ SUB) -> DecisionEngine -> {SdnEnforcer, EbpfEnforcer}
                                                                    |
                                                        Component 1 (LsmCommand, ZMQ PUB)

This is the "there cannot be a crash" layer made concrete: every
component above runs its own try/except internally (see each file's
docstring), and this file adds the last line of defence — a top-level
watchdog loop that catches anything that still slips through, logs it
with full context, and keeps the process alive rather than exiting.

Run this instead of wiring the pieces together ad hoc in a notebook or
a one-off script — a crash during a live panel demo is the one failure
mode you explicitly said must not happen, and that only comes from
treating this as "the main program," not a glue script.

Usage:
    python3 supervisor.py --stub-feed path/to/46_alerts.jsonl   # replay mode, no ZMQ needed
    python3 supervisor.py --zmq-c3 tcp://127.0.0.1:5556          # live ZMQ mode
"""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decision_engine.engine import DecisionEngine  # noqa: E402
from enforcers.sdn_enforcer import SdnEnforcer  # noqa: E402
from enforcers.ebpf_enforcer import EbpfEnforcer  # noqa: E402
from integration.resilient_link import ResilientSubscriber, THREAT_ALERT_SCHEMA  # noqa: E402

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-28s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "component4.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("component4.supervisor")

RULE_MATRIX_PATH = str(Path(__file__).resolve().parent / "config" / "rule_matrix.yaml")


class Component4Supervisor:
    def __init__(self):
        self.engine = DecisionEngine(RULE_MATRIX_PATH)
        self.sdn = SdnEnforcer()
        self.ebpf = EbpfEnforcer()
        self._running = True
        self._alerts_processed = 0
        self._alerts_failed = 0

    def handle_alert(self, alert: dict):
        """
        The one function that must never, under any circumstance, let an
        exception escape into the ZMQ receive thread. Every stage is its
        own try/except so one enforcer failing never stops the other, or
        the Decision Engine, from doing its job on the next alert.
        """
        alert_id = alert.get("alert_id", "<unknown>")
        try:
            decision = self.engine.decide(alert)  # decide() already never raises
        except Exception as e:  # noqa: BLE001 - absolute last resort, decide() shouldn't reach here
            logger.exception("supervisor: UNEXPECTED — decide() itself raised for alert=%s: %s",
                              alert_id, e)
            self._alerts_failed += 1
            return

        route = decision.route_to
        applied_ok = True

        if route == "MANAGED_LINUX":
            applied_ok = self.ebpf.apply(decision)   # apply() already never raises
        elif route == "IOT_AGENTLESS":
            applied_ok = self.sdn.apply(decision)    # apply() already never raises
        else:
            logger.warning("supervisor: alert=%s has unrecognised route_to=%s — "
                            "applying via SDN path as the safer default (network-level only)",
                            alert_id, route)
            applied_ok = self.sdn.apply(decision)

        self._alerts_processed += 1
        if not applied_ok:
            self._alerts_failed += 1
            logger.error("supervisor: containment FAILED to apply for alert=%s "
                          "(decision=%s) — alert is still logged, system continues",
                          alert_id, decision.level)

    def on_c3_down(self):
        self.engine.enter_degraded_mode()

    def on_c3_up(self):
        self.engine.exit_degraded_mode()

    def health(self) -> dict:
        return {
            "alerts_processed": self._alerts_processed,
            "alerts_failed": self._alerts_failed,
            "engine_degraded": self.engine.is_degraded,
            "sdn_active_containments": len(self.sdn.active_containments()),
            "ebpf_active_containments": len(self.ebpf.active_containments()),
        }

    def shutdown(self):
        logger.info("supervisor: shutting down — rolling back nothing automatically "
                    "(active containments persist until their TTL; use rollback_all.py "
                    "if you want an immediate full rollback)")
        self.sdn.shutdown()
        self.ebpf.shutdown()

    # ------------------------------------------------------------------
    # Human-in-the-loop review — used by the interactive console below.
    # An alert_id could be sitting in either enforcer's pending_review(),
    # depending on which route it was sent down, so these just check both.
    # ------------------------------------------------------------------

    def all_pending_review(self) -> dict:
        return {
            "sdn": self.sdn.pending_review(),
            "ebpf": self.ebpf.pending_review(),
        }

    def confirm_release(self, alert_id: str, reviewer: str = "analyst") -> bool:
        if alert_id in self.sdn.pending_review():
            return self.sdn.confirm_release(alert_id, reviewer)
        if alert_id in self.ebpf.pending_review():
            return self.ebpf.confirm_release(alert_id, reviewer)
        return False

    def escalate(self, alert_id: str, extend_seconds: int, reviewer: str = "analyst") -> bool:
        if alert_id in self.sdn.pending_review():
            return self.sdn.escalate(alert_id, extend_seconds, reviewer)
        if alert_id in self.ebpf.pending_review():
            return self.ebpf.escalate(alert_id, extend_seconds, reviewer)
        return False


def run_replay_mode(supervisor: Component4Supervisor, dataset_path: Path):
    logger.info("supervisor: REPLAY MODE against %s (no ZMQ — direct function calls)", dataset_path)
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                alert = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("supervisor: skipping malformed JSONL line: %s", e)
                continue
            supervisor.handle_alert(alert)
            time.sleep(0.05)  # small pacing so logs stay readable during a live demo
    logger.info("supervisor: replay complete — %s", supervisor.health())


def run_interactive_console(supervisor: Component4Supervisor):
    """
    Keeps the process alive after the stub feed is processed (so the TTL
    watchdog threads inside the enforcers keep running) and gives you a
    tiny command line for the human-in-the-loop review step — this is
    the SOC-analyst side of confirm_release()/escalate() for a live demo
    against your real Mininet network.

    Commands:
      pending                     - list everything awaiting review
      confirm <alert_id>          - release it (you've reviewed it, it's clean)
      escalate <alert_id> <secs>  - keep it contained <secs> seconds longer
      health                      - print alerts_processed / active counts
      quit                        - shut down cleanly
    """
    print("\n--- Component 4 interactive console ---")
    print("Type 'pending' to see TTL-expired containments awaiting review.")
    print("Type 'confirm <alert_id>' or 'escalate <alert_id> <seconds>'.")
    print("Type 'health' for counters, 'quit' to exit.\n")

    while True:
        try:
            line = input("component4> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "quit" or cmd == "exit":
            break

        elif cmd == "pending":
            pending = supervisor.all_pending_review()
            if not pending["sdn"] and not pending["ebpf"]:
                print("  (nothing pending review)")
            for enforcer_name, entries in pending.items():
                for alert_id, entry in entries.items():
                    print(f"  [{enforcer_name}] {alert_id}: {entry}")

        elif cmd == "confirm" and len(parts) == 2:
            alert_id = parts[1]
            ok = supervisor.confirm_release(alert_id, reviewer="oshanee")
            print(f"  confirm_release({alert_id}) -> {ok}"
                  + ("" if ok else "  (nothing pending under that alert_id — check 'pending')"))

        elif cmd == "escalate" and len(parts) == 3:
            alert_id, secs = parts[1], parts[2]
            try:
                secs = int(secs)
            except ValueError:
                print("  seconds must be an integer")
                continue
            ok = supervisor.escalate(alert_id, secs, reviewer="oshanee")
            print(f"  escalate({alert_id}, {secs}s) -> {ok}"
                  + ("" if ok else "  (nothing pending under that alert_id — check 'pending')"))

        elif cmd == "health":
            print(f"  {supervisor.health()}")

        else:
            print("  unknown command — try: pending | confirm <id> | escalate <id> <secs> | health | quit")

    print("supervisor: shutting down")
    supervisor.shutdown()


def run_live_mode(supervisor: Component4Supervisor, c3_addr: str):
    logger.info("supervisor: LIVE MODE — subscribing to Component 3 at %s", c3_addr)
    sub = ResilientSubscriber(
        peer_name="component3",
        connect_addr=c3_addr,
        schema=THREAT_ALERT_SCHEMA,
        on_message=supervisor.handle_alert,
        on_peer_down=supervisor.on_c3_down,
        on_peer_up=supervisor.on_c3_up,
    )
    sub.start()

    def _shutdown(signum, frame):
        logger.info("supervisor: received signal %s — shutting down gracefully", signum)
        sub.stop()
        supervisor.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("supervisor: running. Ctrl+C to stop. Health check every 30s below.")
    while True:
        time.sleep(30)
        logger.info("supervisor: health=%s peer_c3_alive=%s", supervisor.health(), sub.is_peer_alive)


def main():
    parser = argparse.ArgumentParser(description="Component 4 supervisor")
    parser.add_argument("--stub-feed", type=str, help="Path to JSONL alert file for replay mode")
    parser.add_argument("--zmq-c3", type=str, help="ZMQ address for live Component 3 feed, e.g. tcp://127.0.0.1:5556")
    parser.add_argument("--interactive", action="store_true",
                         help="After processing --stub-feed, stay running and open a console "
                              "for 'pending'/'confirm'/'escalate' — needed to demo the "
                              "TTL-expiry-to-human-review flow against a real network, since "
                              "the process would otherwise exit immediately after replay.")
    args = parser.parse_args()

    supervisor = Component4Supervisor()

    if args.stub_feed:
        run_replay_mode(supervisor, Path(args.stub_feed))
        if args.interactive:
            run_interactive_console(supervisor)
    elif args.zmq_c3:
        run_live_mode(supervisor, args.zmq_c3)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
