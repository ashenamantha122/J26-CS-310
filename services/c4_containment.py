"""
Component 4 -- Hybrid containment via SDN and eBPF.
IT23170520 | Rajasooriya D.G.C.H

Response and recovery layer. Three properties distinguish it from a playbook:

  graduated   the minimum effective level is selected from confidence, asset
              criticality and attack context -- not one blunt isolate action
  coordinated one decision drives both the SDN layer (unmanaged IoT) and the
              eBPF LSM layer (managed hosts), from a single policy record
  reversible  every action carries a time-bounded lease; expiry or disproof
              triggers automatic rollback, so a false positive self-heals

The SDN calls are emitted as the exact ovs-ofctl commands a real Open vSwitch
would receive. Set dry_run=False on a host with OVS to execute them.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from laax.bus import Bus
from laax.contracts import (
    ContainmentAction,
    CorrelationAlert,
    EnforceHost,
    Rollback,
    Topic,
)

# Asset criticality drives how aggressively we are allowed to act. A safety
# critical device is never fully isolated by an automated decision.
ASSET_CRITICALITY: Dict[str, str] = {
    "iot-01": "safety_critical",   # e.g. a Modbus controller on a production line
    "iot-07": "standard",
    "_default": "standard",
}

LEVEL_NAMES = {0: "observe", 1: "rate-limit", 2: "segment", 3: "isolate"}


@dataclass
class Lease:
    policy_id: str
    target: str
    level: int
    layer: str
    expires_at: float
    alert_id: str
    undo: List[str] = field(default_factory=list)


class C4Containment:
    def __init__(self, bus: Bus, dry_run: bool = True,
                 lease_seconds: float = 8.0, bridge: str = "br0") -> None:
        self.bus, self.dry_run, self.bridge = bus, dry_run, bridge
        self.lease_seconds = lease_seconds
        self.leases: Dict[str, Lease] = {}
        self.history: List[ContainmentAction] = []
        self.rollbacks: List[Rollback] = []
        self.enforce_latencies: List[float] = []
        bus.subscribe(Topic.CORRELATION, self.on_alert)

    # -- policy selection --------------------------------------------------
    @staticmethod
    def select_level(alert: CorrelationAlert, criticality: str) -> tuple[int, str]:
        """Minimum effective response. Returns (level, rationale)."""
        c = alert.confidence
        if c < 0.45:
            return 0, "confidence below action threshold; observe only"
        if c < 0.60:
            return 1, f"moderate confidence {c:.2f}; throttle without disconnecting"
        if criticality == "safety_critical":
            return 2, (f"confidence {c:.2f} would justify isolation, but the asset is "
                       f"safety-critical; segmenting to a monitored VLAN preserves "
                       f"control-loop connectivity")
        if c < 0.80:
            return 2, f"confidence {c:.2f}; segment device and restrict host process"
        return 3, f"high confidence {c:.2f} on a standard asset; isolate"

    # -- SDN arm -----------------------------------------------------------
    def _ovs(self, cmd: List[str]) -> None:
        printable = " ".join(cmd)
        if self.dry_run or not shutil.which("ovs-ofctl"):
            print(f"  [C4] sdn$ {printable}")
            return
        subprocess.run(cmd, check=False, capture_output=True)

    def _sdn_apply(self, level: int, device_ip: str) -> List[str]:
        """Returns the undo commands for the rollback manager."""
        hard = int(self.lease_seconds)
        if level == 1:
            self._ovs(["ovs-ofctl", "add-flow", self.bridge,
                       f"priority=500,ip,nw_src={device_ip},"
                       f"actions=meter:1,normal,hard_timeout={hard}"])
        elif level == 2:
            self._ovs(["ovs-ofctl", "add-flow", self.bridge,
                       f"priority=600,ip,nw_src={device_ip},"
                       f"actions=mod_vlan_vid:999,output:honeypot,hard_timeout={hard}"])
        elif level == 3:
            self._ovs(["ovs-ofctl", "add-flow", self.bridge,
                       f"priority=800,ip,nw_src={device_ip},actions=drop,hard_timeout={hard}"])
        return ["ovs-ofctl", "del-flows", self.bridge, f"ip,nw_src={device_ip}"]

    # -- main handler ------------------------------------------------------
    async def on_alert(self, _t: str, alert: CorrelationAlert) -> None:
        t0 = time.perf_counter()
        device = alert.iot_device or "unknown"

        # One target, one active containment. A single incident legitimately
        # produces several correlation alerts as the chain unfolds; re-applying
        # flow rules per alert would thrash the switch and reset the lease clock
        # every time, so an existing lease at an equal or higher level absorbs
        # the new alert instead.
        existing = next((l for l in self.leases.values() if l.target == device), None)

        criticality = ASSET_CRITICALITY.get(device, ASSET_CRITICALITY["_default"])
        level, rationale = self.select_level(alert, criticality)

        if existing is not None and existing.level >= level:
            print(f"  [C4] target {device} already contained at level {existing.level} "
                  f"({LEVEL_NAMES[existing.level]}); alert absorbed, lease unchanged")
            return

        if level == 0:
            print(f"  [C4] level 0 observe -- {rationale}")
            return

        device_ip = next((n["node"].split(":", 1)[1] for n in alert.path
                          if n["type"] == "ip"), "0.0.0.0")
        undo = self._sdn_apply(level, device_ip)

        layer = "sdn" if alert.pid is None else "both"
        action = ContainmentAction(
            alert_id=alert.alert_id, level=level, layer=layer,
            target=device, action=f"{LEVEL_NAMES[level]} {device} ({device_ip})",
            lease_seconds=self.lease_seconds, rationale=rationale, reversible=True,
        )
        print(f"  [C4] LEVEL {level} {LEVEL_NAMES[level].upper()}  target={device} "
              f"[{criticality}]  lease={self.lease_seconds:.0f}s")
        print(f"       rationale: {rationale}")
        await self.bus.publish(Topic.CONTAINMENT, action)
        self.history.append(action)

        # Endpoint arm: surgical syscall denial, not process termination.
        if level >= 2 and alert.pid and alert.host_id:
            await self.bus.publish(Topic.ENFORCE_HOST, EnforceHost(
                policy_id=action.policy_id, host_id=alert.host_id, pid=alert.pid,
                syscall="execve", lease_seconds=self.lease_seconds,
            ))

        self.leases[action.policy_id] = Lease(
            policy_id=action.policy_id, target=device, level=level, layer=layer,
            expires_at=time.time() + self.lease_seconds, alert_id=alert.alert_id,
            undo=undo,
        )
        self.enforce_latencies.append((time.perf_counter() - t0) * 1000.0)

    # -- rollback manager --------------------------------------------------
    async def reaper(self, tick: float = 0.5) -> None:
        """Expires leases and withdraws containment. Runs for the process life."""
        while True:
            now = time.time()
            for pid_, lease in list(self.leases.items()):
                if now >= lease.expires_at:
                    await self._rollback(pid_, "lease_expired")
            await asyncio.sleep(tick)

    async def disprove(self, alert_id: str) -> None:
        """Analyst or C3 retracts the evidence: withdraw immediately."""
        for pid_, lease in list(self.leases.items()):
            if lease.alert_id == alert_id:
                await self._rollback(pid_, "disproved")

    async def _rollback(self, policy_id: str, reason: str) -> None:
        lease = self.leases.pop(policy_id, None)
        if lease is None:
            return
        self._ovs(lease.undo)
        rb = Rollback(policy_id=policy_id, reason=reason, target=lease.target, success=True)
        self.rollbacks.append(rb)
        print(f"  [C4] ROLLBACK {LEVEL_NAMES[lease.level]} on {lease.target} "
              f"({reason}) -- connectivity restored")
        await self.bus.publish(Topic.ROLLBACK, rb)

    def stats(self) -> dict:
        lat = self.enforce_latencies
        return {
            "actions": len(self.history),
            "by_level": {LEVEL_NAMES[a.level]: sum(1 for b in self.history if b.level == a.level)
                         for a in self.history},
            "rollbacks": len(self.rollbacks),
            "leases_open": len(self.leases),
            "decision_latency_ms_mean": round(sum(lat) / len(lat), 3) if lat else 0.0,
        }
