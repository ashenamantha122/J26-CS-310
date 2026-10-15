"""
Component 3 -- Cross-domain threat correlation engine.
IT23187832 | Peramunugama M.R.A

This is the integration hub. It consumes HostEvent from C1 and FlowVerdict from
C2, maintains one multi-partite graph over both domains, learns what normal
structure looks like, and reports connected paths that neither domain could
reveal alone.

    iot_device --flow--> ip --socket--> process --spawn--> process

Two signals are combined, deliberately:
  structural : a GNN autoencoder trained on benign snapshots. High
               reconstruction error means this node sits in a shape the network
               has not exhibited before.
  path       : a connected route from an IoT device C2 scored malicious to a
               managed host process, inside a time window.

Either alone produces noise. The conjunction is the contribution.

The GNN is a mean-aggregating GraphSAGE-style encoder written directly in
torch, so the prototype needs no torch_geometric. Swapping in PyG later means
replacing TinyGraphSAGE and nothing else.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from laax.bus import Bus
from laax.contracts import (
    CorrelationAlert,
    ExternalAlert,
    FlowVerdict,
    HostEvent,
    Topic,
)

NODE_TYPES = ["iot_device", "ip", "socket", "process", "host"]
FEAT_DIM = len(NODE_TYPES) + 4          # type one-hot + 4 structural features


# --------------------------------------------------------------------------
class TinyGraphSAGE(nn.Module):
    """Two-layer mean-aggregating encoder + decoder, trained as an autoencoder."""

    def __init__(self, in_dim: int = FEAT_DIM, hid: int = 16, emb: int = 8) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim * 2, hid)
        self.l2 = nn.Linear(hid * 2, emb)
        self.dec = nn.Sequential(nn.Linear(emb, hid), nn.ReLU(), nn.Linear(hid, in_dim))

    @staticmethod
    def _aggregate(x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (adj @ x) / deg

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.l1(torch.cat([x, self._aggregate(x, adj)], dim=1)))
        return self.l2(torch.cat([h, self._aggregate(h, adj)], dim=1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.dec(self.encode(x, adj))


# --------------------------------------------------------------------------
class CrossDomainGraph:
    """The multi-partite graph, plus featurisation."""

    def __init__(self, window_s: float = 120.0) -> None:
        self.g = nx.DiGraph()
        self.window_s = window_s

    def _touch(self, node: str, ntype: str, **attrs) -> None:
        if node not in self.g:
            self.g.add_node(node, ntype=ntype, first_seen=time.time(), score=0.0)
        self.g.nodes[node].update(last_seen=time.time(), **attrs)

    def add_host_event(self, ev: HostEvent) -> None:
        proc = f"proc:{ev.host_id}:{ev.pid}"
        self._touch(proc, "process", comm=ev.comm, pid=ev.pid, host_id=ev.host_id,
                    confidence=ev.confidence, depth=ev.depth)
        self._touch(f"host:{ev.host_id}", "host")
        self.g.add_edge(f"host:{ev.host_id}", proc, kind="runs")
        if ev.ppid:
            parent = f"proc:{ev.host_id}:{ev.ppid}"
            self._touch(parent, "process", pid=ev.ppid, host_id=ev.host_id)
            self.g.add_edge(parent, proc, kind="spawn")
        if ev.kind in ("connect", "accept") and ev.daddr:
            sock = f"sock:{ev.daddr}:{ev.dport}"
            self._touch(sock, "socket", dport=ev.dport)
            self._touch(f"ip:{ev.daddr}", "ip")
            # Direction matters: the IoT side reaches the process through the
            # socket, so the path runs ip -> socket -> process.
            self.g.add_edge(f"ip:{ev.daddr}", sock, kind="endpoint")
            self.g.add_edge(sock, proc, kind="socket")
            if ev.saddr:
                self._touch(f"ip:{ev.saddr}", "ip")
                self.g.add_edge(f"ip:{ev.saddr}", sock, kind="peer")

    def add_flow_verdict(self, fv: FlowVerdict) -> None:
        dev = f"iot:{fv.device_id}"
        self._touch(dev, "iot_device", verdict=fv.verdict, score=fv.score)
        self._touch(f"ip:{fv.saddr}", "ip")
        self.g.add_edge(dev, f"ip:{fv.saddr}", kind="owns")
        self._touch(f"ip:{fv.daddr}", "ip")
        self.g.add_edge(f"ip:{fv.saddr}", f"ip:{fv.daddr}", kind="flow", score=fv.score)

    # -- featurisation -----------------------------------------------------
    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        nodes = list(self.g.nodes)
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        x = np.zeros((n, FEAT_DIM), dtype=np.float32)
        now = time.time()
        for node in nodes:
            i, d = idx[node], self.g.nodes[node]
            x[i, NODE_TYPES.index(d["ntype"])] = 1.0
            x[i, len(NODE_TYPES) + 0] = np.log1p(self.g.in_degree(node))
            x[i, len(NODE_TYPES) + 1] = np.log1p(self.g.out_degree(node))
            x[i, len(NODE_TYPES) + 2] = float(d.get("score", 0.0))
            x[i, len(NODE_TYPES) + 3] = np.exp(-(now - d.get("last_seen", now)) / self.window_s)
        adj = np.zeros((n, n), dtype=np.float32)
        for u, v in self.g.edges:
            adj[idx[v], idx[u]] = 1.0        # aggregate from predecessors
            adj[idx[u], idx[v]] = 1.0        # and successors: structure is undirected
        return torch.from_numpy(x), torch.from_numpy(adj), nodes


# --------------------------------------------------------------------------
class C3Correlator:
    def __init__(self, bus: Bus, alert_threshold: float = 0.55,
                 boost: float = 6.0) -> None:
        self.bus = bus
        self.graph = CrossDomainGraph()
        self.model = TinyGraphSAGE()
        self.trained = False
        self.alert_threshold = alert_threshold
        self.boost = boost
        self.suspect_ips: Dict[str, float] = {}     # ip -> C2 score
        self.flagged_at: Dict[str, float] = {}      # ip -> when C2 first flagged it
        self.alerted: set = set()
        self.alerts: List[CorrelationAlert] = []
        bus.subscribe(Topic.HOST_EVENT, self.on_host_event)
        bus.subscribe(Topic.FLOW_VERDICT, self.on_flow_verdict)

    # -- ingest ------------------------------------------------------------
    async def on_host_event(self, _t: str, ev: HostEvent) -> None:
        self.graph.add_host_event(ev)
        await self._evaluate()

    async def on_flow_verdict(self, _t: str, fv: FlowVerdict) -> None:
        self.graph.add_flow_verdict(fv)
        if fv.verdict in ("suspicious", "malicious"):
            self.suspect_ips[fv.saddr] = fv.score
            self.flagged_at.setdefault(fv.saddr, time.time())
        await self._evaluate()

    # -- structural learning ----------------------------------------------
    def train_baseline(self, epochs: int = 220) -> float:
        """Fit the autoencoder on the current (benign) graph snapshot."""
        x, adj, _ = self.graph.tensors()
        if x.shape[0] < 4:
            return float("nan")
        opt = torch.optim.Adam(self.model.parameters(), lr=0.02)
        loss_fn = nn.MSELoss()
        self.model.train()
        loss = torch.tensor(0.0)
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.model(x, adj), x)
            loss.backward()
            opt.step()
        self.model.eval()
        self.trained = True
        return float(loss.item())

    def structural_scores(self) -> Dict[str, float]:
        x, adj, nodes = self.graph.tensors()
        if not self.trained or x.shape[0] < 4:
            return {n: 0.0 for n in nodes}
        with torch.no_grad():
            err = ((self.model(x, adj) - x) ** 2).mean(dim=1)
        e = err.numpy()
        rng = float(e.max() - e.min())
        norm = (e - e.min()) / rng if rng > 1e-9 else np.zeros_like(e)
        return {n: float(v) for n, v in zip(nodes, norm)}

    # -- path search -------------------------------------------------------
    #
    # A note on why this is not a generic shortest_path search.
    #
    # The graph contains a host node joined to every process it runs. That edge
    # is an ownership annotation, not a communication channel, so a naive
    # undirected shortest path hops attacker -> socket -> any process -> host ->
    # every other process, and every process on the box looks compromised. The
    # traversal below only follows edges that represent an actual causal route:
    # the attacker reaches an entry process through a socket, and the incident
    # then propagates along process spawn edges.
    ENTRY_EDGES = {"endpoint", "peer", "socket"}

    def _incidents(self, max_descendants: int = 12) -> List[Tuple[str, List[str], List[str]]]:
        """
        Returns (iot_device, entry_path, descendant_pids) per suspect entry.

        entry_path is [iot, ip, socket, entry_process]; descendants are the
        processes spawned beneath the entry process, i.e. the pivot chain.
        """
        g = self.graph.g
        out: List[Tuple[str, List[str], List[str]]] = []
        for ip in self.suspect_ips:
            src = f"ip:{ip}"
            if src not in g:
                continue
            owner = next((n for n in g.predecessors(src)
                          if g.nodes[n]["ntype"] == "iot_device"), None)
            # Only processes that appeared AFTER the device was flagged can be
            # part of this incident. Without this bound, every child a
            # long-lived entry process ever spawned -- an sshd with a month of
            # legitimate login sessions beneath it -- is swept into the chain.
            t_flag = self.flagged_at.get(ip, 0.0)
            # ip -> socket -> process, following only entry-kind edges
            for sock in g.successors(src):
                if g.nodes[sock]["ntype"] != "socket":
                    continue
                for proc in g.successors(sock):
                    if g.nodes[proc]["ntype"] != "process":
                        continue
                    entry_path = ([owner] if owner else []) + [src, sock, proc]
                    # Walk spawn edges downward to recover the pivot chain.
                    descendants: List[str] = []
                    frontier = [proc]
                    seen = {proc}
                    while frontier and len(descendants) < max_descendants:
                        cur = frontier.pop(0)
                        for nxt in g.successors(cur):
                            if nxt in seen or g.nodes[nxt]["ntype"] != "process":
                                continue
                            if g.edges[cur, nxt].get("kind") != "spawn":
                                continue
                            if g.nodes[nxt].get("first_seen", 0.0) < t_flag:
                                continue          # predates the incident
                            seen.add(nxt)
                            descendants.append(nxt)
                            frontier.append(nxt)
                    out.append((owner, entry_path, descendants))
        return out

    # -- decision ----------------------------------------------------------
    async def _evaluate(self) -> None:
        if not self.suspect_ips:
            return
        struct = self.structural_scores()
        for owner, entry_path, descendants in self._incidents():
            entry = entry_path[-1]
            if entry in self.alerted:
                continue
            pdata = self.graph.g.nodes[entry]
            iot_score = max(self.suspect_ips.values())
            struct_score = max([struct.get(n, 0.0) for n in entry_path + descendants]
                               or [0.0])
            confidence = 0.6 * iot_score + 0.4 * struct_score
            if confidence < self.alert_threshold:
                continue
            self.alerted.add(entry)

            full = entry_path + descendants
            alert = CorrelationAlert(
                severity="high" if confidence > 0.75 else "medium",
                confidence=round(confidence, 4),
                path=[{"node": n, "type": self.graph.g.nodes[n]["ntype"],
                       "comm": self.graph.g.nodes[n].get("comm")} for n in full],
                iot_device=owner.split(":", 1)[1] if owner else None,
                host_id=pdata.get("host_id"),
                pid=pdata.get("pid"),
                explanation=(
                    f"IoT flow score {iot_score:.2f} reaches {pdata.get('comm', '?')} "
                    f"(pid {pdata.get('pid')}) via socket; {len(descendants)} spawned "
                    f"descendants follow; structural novelty {struct_score:.2f}"
                ),
            )
            self.alerts.append(alert)
            chain = " -> ".join(
                self.graph.g.nodes[n].get("comm") or n.split(":", 1)[0] for n in full)
            print(f"  [C3] CORRELATED  conf={confidence:.3f}")
            print(f"       chain: {chain}")
            print(f"       {alert.explanation}")

            # Fan out: containment to C4, and E(t) back to C1. The boost goes to
            # the entry process AND its descendants, because the pivot chain is
            # what needs forensic depth, not just the process that accepted the
            # connection.
            await self.bus.publish(Topic.CORRELATION, alert)
            for node in [entry] + descendants:
                d = self.graph.g.nodes[node]
                if d.get("host_id") and d.get("pid"):
                    await self.bus.publish(Topic.EXTERNAL_ALERT, ExternalAlert(
                        host_id=d["host_id"], pid=d["pid"], boost=self.boost,
                        reason=f"cross-domain path from {alert.iot_device}",
                        alert_id=alert.alert_id,
                    ))

    def stats(self) -> dict:
        types = defaultdict(int)
        for _, d in self.graph.g.nodes(data=True):
            types[d["ntype"]] += 1
        return {
            "nodes": self.graph.g.number_of_nodes(),
            "edges": self.graph.g.number_of_edges(),
            "by_type": dict(types),
            "alerts": len(self.alerts),
            "gnn_trained": self.trained,
        }
