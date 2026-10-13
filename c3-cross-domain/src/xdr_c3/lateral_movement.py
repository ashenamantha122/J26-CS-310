"""Function 2 - evidence-based identification of POTENTIAL lateral movement in F1-suspicious scenarios.

This is deterministic rule-based verification, not a model. It consumes `f1_f2_handoff.ScenarioHandoff` records and
never recomputes a score, touches the threshold, reads ground truth or uses reconstruction errors, anomaly scores or
the Component 1/2 predictions (`threat_state`, `predicted_class`, ...) to decide anything.

What the data genuinely provides (verified on the real data before these rules were written)
  * Direction of a host-to-host communication: a Stage 3 `connect_accept` link joins a `network_connect` on host A
    (src_ip == A's host_ip) to a `network_accept` on host B (dst_ip == B's host_ip) with an identical
    (src ip, dst ip, src port, dst port, protocol) tuple, distinct host_ip and host_id. Direction therefore comes from
    the event fields, NOT from the Stage 4 graph (whose edges are stored in both directions for message passing).
  * Original link orientation: `evidence.node_a -> node_b`. For process links this is "earlier -> later" by
    construction in Stage 3, but each order is re-verified here from the original timestamps.
  * Timestamps have 1-second resolution: the accept of a connect/accept pair is 0 or 1 s after the connect (never
    before), so equal timestamps inside ONE pair are normal and carry no ordering claim; direction does not depend on it.
  * Process context: host_id/host_ip, process_id, parent_process_id, process_name, event_type, ports, protocol.
What it cannot establish: intent, maliciousness, or that a legitimate proxy/API chain is not what was observed. The
Stage 3b audit found legitimate multi-hop chains with the same shape; `pivot_kind` and the limitations say so.

Assessments (the second is deliberately a weaker statement than the first):
  * `potential_lateral_movement_supported` - at least one verified path that contains a PROCESS-LINEAGE pivot.
  * `cross_host_service_chain_only` - verified cross-host progression(s), but every pivot is a service hand-off
    (the same long-lived service process accepts, then connects onward). Informational: this is ordinary
    application-tier behaviour (a reverse proxy calling an API, an app calling its cache/database) and it is as common
    among scenarios F1 calls Normal as among suspicious ones. It never qualifies as potential lateral movement.
  * `no_supporting_path` - no qualifying path WITHIN THE AVAILABLE SCENARIO EVIDENCE. It is not proof that lateral
    movement did not occur: part of a chain can lie outside the scenario (Stage 3b splits some chains), and anomalies
    unrelated to movement are also reported here.
  * `insufficient_evidence` - too little, ambiguous, inconsistent or unsearched evidence for a reliable conclusion.

A path qualifies for `potential_lateral_movement_supported` when it has all of:
  1. an inbound `network_accept` on host A (the entry event);
  2. a chain of >= 1 Stage 3/3b host-internal link on host A, each re-verified against the event fields, with STRICTLY
     increasing timestamps, ending in a `network_connect` on host A, and containing at least one process-lineage link
     (entry_to_child, parent_child, same_process or fallback_accept). A chain made only of `service_handoff` links does
     not qualify;
  3. a verified directed `connect_accept` hop from that connect to a `network_accept` on a different host B
     (distinct host_ip and host_id, tuple-consistent, accept not before connect, within the Stage 3 tolerance of 5 s);
  4. no repeated event and no repeated host along the path; the host the entry connection came from (the entry's
     source address) counts as already visited, so a route cannot simply return to its origin.
So >= 3 events on >= 2 hosts. A host-to-host connection alone, a single unusual edge, a high score or a connected
graph path never suffices. A process-lineage path can still be ordinary administration (for example SSH hopping).
Further hops (B -> C ...) extend the path only through the same verified segments.
Equal timestamps between different events of the chain, links that fail re-verification, or an exhausted search
bound leave the question open (`insufficient_evidence`); strictly inverted chronology or a contradicted direction
rejects the candidate (it counts toward `no_supporting_path`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .f1_f2_handoff import F1F2Handoff, ScenarioHandoff

SUPPORTED = "potential_lateral_movement_supported"
NO_PATH = "no_supporting_path"
SERVICE_CHAIN_ONLY = "cross_host_service_chain_only"
INSUFFICIENT = "insufficient_evidence"
ASSESSMENT_MEANING = {
    SUPPORTED: "a verified, directed, time-consistent cross-host progression with a process-lineage pivot exists in this "
               "scenario; potential movement only - intent and maliciousness are not established",
    SERVICE_CHAIN_ONLY: "informational: a verified cross-host chain exists, but every pivot is a service hand-off "
                        "(ordinary application-service behaviour); not potential lateral movement",
    NO_PATH: "no qualifying path was found within the available scenario evidence; this is not proof that lateral "
             "movement did not occur",
    INSUFFICIENT: "the available evidence cannot support a reliable conclusion",
}

MIN_EVENTS = 3                  # entry accept + connect + remote accept
MAX_PIVOT_EVENTS = 12           # same bound as the lineage depth used in the Stage 3b audit (12)
MAX_HOPS = 6                    # cross-host hops along one path
MAX_CANDIDATES = 500            # per-scenario enumeration cap; reaching it is reported, never hidden
HOP_TOLERANCE_S = 5             # Stage 3 HOST_HOST_TOL: connect <-> accept within 5 s

LINEAGE_LINKS = ("entry_to_child", "parent_child", "same_process", "fallback_accept")
INTERNAL_LINKS = ("entry_to_child", "parent_child", "same_process", "fallback_accept", "service_handoff")
HOP_LINK = "connect_accept"
LINK_CATEGORY = {"iot_host_match": "iot_to_host", "connect_accept": "host_to_host",
                 "entry_to_child": "host_internal_process", "parent_child": "host_internal_process",
                 "same_process": "host_internal_process", "fallback_accept": "host_internal_process",
                 "service_handoff": "host_internal_process", "same_remote_source": "host_internal_remote_source",
                 "flow_same_src_dst": "iot_internal_flow", "flow_same_src": "iot_internal_flow",
                 "flow_same_dst": "iot_internal_flow"}
TUPLE = ("src_ip", "dst_ip", "src_port", "dst_port", "protocol")


class NotSuspiciousError(ValueError):
    """Function 2 only accepts scenarios the saved F1 model classified Suspicious."""


class F1GateError(ValueError):
    """A handoff record contradicts the authoritative F1 results (`handoff.f1_scores`)."""

    def __init__(self, problems):
        self.problems = problems
        super().__init__(f"{len(problems)} F1 gate violation(s): " + "; ".join(problems[:5]))


# --------------------------------------------------------------------------- result objects
@dataclass
class LateralMovementAssessment:
    scenario_id: int
    f1: dict                         # carried unchanged from the handoff (score, threshold, prediction, ...)
    assessment: str                  # SUPPORTED | NO_PATH | INSUFFICIENT
    reason_code: str
    paths: list                      # candidate paths (dicts), supported first; see `_path_dict`
    involved: dict                   # hosts / processes / events of the supported paths (empty otherwise)
    relationship_summary: dict       # observed relationship categories in the scenario
    reasoning: list                  # human-readable facts that support / limit the assessment
    limitations: list
    evidence_problems: list
    search: dict                     # bounds and counts of the traversal
    flags: list = field(default_factory=list)   # observations that do not change the assessment (see analyse_scenario)
    evidence_policy: dict = field(default_factory=lambda: {
        "decision_inputs": ["host/IoT telemetry fields", "Stage 3/3b link evidence (re-verified)", "original timestamps"],
        "not_used_for_decision": ["anomaly score", "reconstruction errors", "component predictions (C1/C2)",
                                  "ground-truth labels"],
        "claim": "potential structural cross-host progression only; intent and maliciousness are not established"})

    @property
    def supported_paths(self):
        """Every fully verified path (process-lineage and service-chain-only alike)."""
        return [p for p in self.paths if p["status"] == "supported"]

    @property
    def qualifying_paths(self):
        return [p for p in self.supported_paths if p["qualifies"]]

    @property
    def service_chain_paths(self):
        return [p for p in self.supported_paths if not p["qualifies"]]

    @property
    def evidence_types(self):
        """Pivot evidence types present among the verified paths (both are recorded when both exist)."""
        return sorted({t for p in self.supported_paths for t in p["evidence_types"]})

    @property
    def assessment_meaning(self):
        return ASSESSMENT_MEANING[self.assessment]

    def to_dict(self):
        return {"scenario_id": int(self.scenario_id), "f1": self.f1, "assessment": self.assessment,
                "reason_code": self.reason_code, "paths": self.paths, "involved": self.involved,
                "relationship_summary": self.relationship_summary, "reasoning": self.reasoning,
                "limitations": self.limitations, "evidence_problems": self.evidence_problems, "search": self.search,
                "flags": self.flags, "evidence_types": self.evidence_types,
                "qualifying_path_ids": [p["path_id"] for p in self.qualifying_paths],
                "service_chain_path_ids": [p["path_id"] for p in self.service_chain_paths],
                "assessment_meaning": self.assessment_meaning, "evidence_policy": self.evidence_policy}


@dataclass
class F2Report:
    assessments: dict                # scenario_id -> LateralMovementAssessment
    errors: dict                     # scenario_id -> processing error text (never silently dropped)

    def counts(self):
        c = {SUPPORTED: 0, SERVICE_CHAIN_ONLY: 0, NO_PATH: 0, INSUFFICIENT: 0}
        for a in self.assessments.values():
            c[a.assessment] += 1
        return {"analysed": len(self.assessments) + len(self.errors), **c, "processing_errors": len(self.errors)}


# --------------------------------------------------------------------------- event access and link verification
def _events(rec: ScenarioHandoff):
    ev = {}
    for kind, frame in (("host", rec.host_events), ("iot", rec.iot_events)):
        cols = list(frame.columns)
        for tup in frame.itertuples(index=False):
            d = dict(zip(cols, tup))
            d["kind"] = kind
            ev[int(d["local_index"])] = d
    nd = rec.nodes.set_index("local_index")
    for li, d in ev.items():
        d["node"] = nd.at[li, "node"]
        d["ts_s"] = int(nd.at[li, "ts_s"])
        d["timestamp"] = nd.at[li, "timestamp"]
    return ev


def verify_link(link_type, a, b):
    """Re-verify one Stage 3/3b link against the original event fields (not trusting the link table).
    Returns (ok, detail). `a` / `b` are event dicts in the ORIGINAL link orientation."""
    host_ev = a["kind"] == "host" and b["kind"] == "host"
    same_host = host_ev and a["host_ip"] == b["host_ip"] and a["host_id"] == b["host_id"]
    if link_type == HOP_LINK:
        if not host_ev:
            return False, "connect_accept must join two host events"
        if a["event_type"] != "network_connect" or b["event_type"] != "network_accept":
            return False, "orientation is not connect -> accept"
        if a["src_ip"] != a["host_ip"] or b["dst_ip"] != b["host_ip"] or a["dst_ip"] != b["host_ip"]:
            return False, "connect source / accept destination do not match the hosts' own addresses"
        if a["host_ip"] == b["host_ip"] or a["host_id"] == b["host_id"]:
            return False, "connect and accept are on the same host"
        if any(a[k] != b[k] for k in TUPLE):
            return False, "network tuples differ"
        return True, "connect on %s -> accept on %s, identical tuple" % (a["host_id"], b["host_id"])
    if link_type == "iot_host_match":
        if a["kind"] != "iot" or b["kind"] != "host" or b["event_type"] not in ("network_connect", "network_accept"):
            return False, "iot_host_match must join an IoT flow to a host network event"
        direct = all(a[k] == b[k] for k in TUPLE)
        rev = (a["src_ip"] == b["dst_ip"] and a["dst_ip"] == b["src_ip"] and a["src_port"] == b["dst_port"]
               and a["dst_port"] == b["src_port"] and a["protocol"] == b["protocol"])
        if not (direct or rev):
            return False, "IoT flow tuple does not match the host event tuple"
        return True, "tuple match (%s)" % ("direct" if direct else "reversed")
    if link_type not in INTERNAL_LINKS:
        return True, "context link (not used for movement)"
    if not same_host:
        return False, "internal link joins different hosts or non-host events"
    if link_type == "entry_to_child":
        ok = (a["event_type"] == "network_accept" and b["event_type"] == "process_start"
              and b["parent_process_id"] == a["process_id"])
        return ok, "accept -> child process started by the accepting process" if ok else "parent/child fields do not match"
    if link_type == "parent_child":
        ok = b["event_type"] == "process_start" and b["parent_process_id"] == a["process_id"]
        return ok, "event of parent process -> child process_start" if ok else "parent/child fields do not match"
    if link_type == "same_process":
        ok = a["process_id"] == b["process_id"]
        return ok, "consecutive events of the same process" if ok else "process ids differ"
    if link_type == "fallback_accept":
        ok = a["event_type"] == "network_accept" and b["event_type"] == "process_start"
        return ok, "accept -> process_start (parent pid has no row; pid relation not verifiable)" if ok else "event types do not match"
    if link_type == "service_handoff":
        ok = (a["event_type"] == "network_accept" and b["event_type"] == "network_connect"
              and a["process_id"] == b["process_id"])
        return ok, "service process accepts then connects onward" if ok else "event types / process ids do not match"
    return False, "unknown link type"


def _order(a, b):
    d = b["ts_s"] - a["ts_s"]
    return "ordered" if d > 0 else ("equal_timestamp" if d == 0 else "inverted")


def _ref(rec, row):
    return {"scenario_id": rec.scenario_id, "edge_index": None if pd.isna(row.edge_index) else int(row.edge_index),
            "node_a": row.node_a, "node_b": row.node_b, "link_type": row.link_type, "points": int(row.points),
            "gap_sec": float(row.gap_sec)}


def _event_view(e):
    out = {"event": e["node"], "local_index": int(e["local_index"]), "timestamp": e["timestamp"], "ts_s": e["ts_s"],
           "kind": e["kind"]}
    if e["kind"] == "host":
        out.update({k: e[k] for k in ("host_id", "host_ip", "process_id", "parent_process_id", "process_name",
                                      "event_type", "src_ip", "dst_ip", "src_port", "dst_port", "protocol")})
        out["process_id"], out["parent_process_id"] = int(out["process_id"]), int(out["parent_process_id"])
    else:
        out.update({k: e[k] for k in ("iot_device_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "service")})
    return out


# --------------------------------------------------------------------------- analysis
def analyse_scenario(rec: ScenarioHandoff, max_candidates=MAX_CANDIDATES) -> LateralMovementAssessment:
    f1 = rec.f1
    if f1.get("prediction") != "Suspicious" or not f1.get("anomaly_score", 0) > f1.get("threshold", float("inf")):
        raise NotSuspiciousError(f"scenario {rec.scenario_id} was not classified Suspicious by F1; Function 2 does not analyse it")
    reasoning, limitations, problems = [], [], []
    base = dict(scenario_id=rec.scenario_id, f1=dict(f1))
    search = {"entries_examined": 0, "candidates": 0, "max_candidates": max_candidates, "truncated": False,
              "host_revisits_rejected": 0, "origin_revisits_rejected": 0,
              "bounds": {"min_events": MIN_EVENTS, "max_pivot_events": MAX_PIVOT_EVENTS, "max_hops": MAX_HOPS}}

    def done(assessment, code, paths=(), involved=None, rel=None, flags=()):
        return LateralMovementAssessment(
            assessment=assessment, reason_code=code, paths=list(paths), involved=involved or {},
            relationship_summary=rel or {}, reasoning=reasoning, limitations=limitations, evidence_problems=problems,
            search=search, flags=list(flags), **base)

    if not rec.f2_ready:
        problems.extend(rec.integrity_problems)
        reasoning.append("the handoff record failed its integrity checks, so its evidence cannot be relied on")
        return done(INSUFFICIENT, "handoff_integrity_problem")

    ev = _events(rec)
    n = len(ev)
    ledger = rec.evidence.drop_duplicates(["node_a", "node_b", "link_type"]).reset_index(drop=True)
    dup = len(rec.evidence) - len(ledger)
    if dup:
        limitations.append(f"{dup} duplicate relationship row(s) were collapsed")
    to_local = dict(zip(rec.nodes["node"], rec.nodes["local_index"]))
    rel_summary = {}
    for lt in ledger["link_type"]:
        cat = LINK_CATEGORY.get(lt, "other")
        rel_summary[cat] = rel_summary.get(cat, 0) + 1
    rel_summary = dict(sorted(rel_summary.items()))

    # verified, original-orientation link tables
    internal, hops, iot_entry = {}, {}, {}
    def sort_key(r):                                    # tolerant of links that point outside the scenario
        a_, b_ = to_local.get(r.node_a), to_local.get(r.node_b)
        return (ev[a_]["ts_s"] if a_ is not None else -1, -1 if a_ is None else a_, -1 if b_ is None else b_, r.link_type)

    for row in sorted(ledger.itertuples(index=False), key=sort_key):
        a, b = to_local.get(row.node_a), to_local.get(row.node_b)
        if a is None or b is None:
            problems.append(f"link {row.link_type} {row.node_a}->{row.node_b} references an event outside the scenario")
            continue
        ok, detail = verify_link(row.link_type, ev[a], ev[b])
        if not ok:
            problems.append(f"link {row.link_type} {row.node_a}->{row.node_b} failed re-verification: {detail}")
            continue
        if row.link_type in INTERNAL_LINKS:
            internal.setdefault(a, []).append((b, row, detail))
        elif row.link_type == HOP_LINK:
            hops.setdefault(a, []).append((b, row, detail))
        elif row.link_type == "iot_host_match":
            iot_entry.setdefault(b, []).append((a, row, detail))

    if n < MIN_EVENTS:
        reasoning.append(f"the scenario has only {n} event(s); a progression needs an inbound event, a pivot and a remote "
                         f"accept (>= {MIN_EVENTS} events), so no reliable conclusion is possible")
        return done(INSUFFICIENT, "fewer_than_3_events", rel=rel_summary)

    n_hosts = len({e["host_ip"] for e in ev.values() if e["kind"] == "host"})
    n_hops_total = sum(len(v) for v in hops.values())
    entries = sorted((li for li, e in ev.items() if e["kind"] == "host" and e["event_type"] == "network_accept"),
                     key=lambda li: (ev[li]["ts_s"], li))
    candidates = []
    chain_cache = {}

    def pivot_chains(entry):
        """All simple chains of verified internal links from `entry` reaching a connect that has a hop."""
        if entry in chain_cache:
            return chain_cache[entry]
        out = chain_cache.setdefault(entry, [])

        def walk(cur, chain, seen):
            if len(chain) + 1 > MAX_PIVOT_EVENTS:
                return
            for (nxt, row, detail) in internal.get(cur, []):
                if nxt in seen:
                    continue
                step = (cur, nxt, row, detail)
                if ev[nxt]["event_type"] == "network_connect" and nxt in hops:
                    out.append(chain + [step])
                walk(nxt, chain + [step], seen | {nxt})

        walk(entry, [], {entry})
        return out

    def extend(entry, segments, used, hosts_ips, hosts_ids):
        """Depth-first, deterministic. Returns True if a fully verified extension was recorded."""
        search["entries_examined"] += 1
        any_verified = False
        for chain in pivot_chains(entry):
            c = chain[-1][1]
            for (p, hrow, hdetail) in hops[c]:
                if search["candidates"] >= max_candidates:
                    search["truncated"] = True
                    return any_verified
                search["candidates"] += 1
                events_used = {entry} | {s[1] for s in chain} | {p}
                seg_reasons, status = [], "verified"
                if events_used & used:
                    continue                                    # repeated event: not a path
                if ev[p]["host_ip"] in hosts_ips or ev[p]["host_id"] in hosts_ids:
                    search["host_revisits_rejected"] += 1       # repeated host (incl. the origin): not a new host
                    if ev[p]["host_ip"] == (segments[0]["entry"]["src_ip"] if segments else ev[entry]["src_ip"]):
                        search["origin_revisits_rejected"] += 1
                    continue
                steps = []
                for (u, v, row, detail) in chain:
                    order = _order(ev[u], ev[v])
                    steps.append({"link_type": row.link_type, "from_event": ev[u]["node"], "to_event": ev[v]["node"],
                                  "from_timestamp": ev[u]["timestamp"], "to_timestamp": ev[v]["timestamp"],
                                  "order": order, "check": detail, "evidence_ref": _ref(rec, row)})
                    if order == "inverted":
                        status = "rejected"
                        seg_reasons.append(f"{row.link_type} {ev[u]['node']} -> {ev[v]['node']}: later event has an earlier timestamp")
                    elif order == "equal_timestamp" and status != "rejected":
                        status = "ambiguous"
                        seg_reasons.append(f"{row.link_type} {ev[u]['node']} -> {ev[v]['node']}: equal timestamps, order unresolved")
                dts = ev[p]["ts_s"] - ev[c]["ts_s"]
                if dts < 0:
                    status = "rejected"
                    seg_reasons.append("accept is timestamped before its connect")
                elif dts > HOP_TOLERANCE_S:
                    status = "rejected"
                    seg_reasons.append("accept is later than the Stage 3 tolerance")
                hop = {"link_type": HOP_LINK, "connect_event": ev[c]["node"], "accept_event": ev[p]["node"],
                       "source": {"host_id": ev[c]["host_id"], "host_ip": ev[c]["host_ip"], "port": ev[c]["src_port"]},
                       "destination": {"host_id": ev[p]["host_id"], "host_ip": ev[p]["host_ip"], "port": ev[p]["dst_port"]},
                       "protocol": ev[c]["protocol"], "direction": "connect_host -> accept_host (from event fields)",
                       "accept_minus_connect_s": int(dts),
                       "time_order": "same_second (not resolvable at 1 s resolution; direction from fields)" if dts == 0
                       else ("connect_before_accept" if dts > 0 else "accept_before_connect"),
                       "check": hdetail, "evidence_ref": _ref(rec, hrow)}
                link_kinds = {s["link_type"] for s in steps}
                pivot_kind = ("service_handoff" if link_kinds == {"service_handoff"}
                              else ("mixed" if "service_handoff" in link_kinds else "process_lineage"))
                entry_src = _entry_context(entry, ev, iot_entry, rec)
                seg = {"entry": _event_view(ev[entry]), "entry_context": entry_src, "pivot_kind": pivot_kind,
                       "pivot_links": steps, "connect": _event_view(ev[c]), "hop": hop, "remote_accept": _event_view(ev[p]),
                       "status": status, "reasons": seg_reasons}
                path_segments = segments + [seg]
                if status == "verified":
                    deeper = False
                    if len(path_segments) < MAX_HOPS:
                        deeper = extend(p, path_segments, used | (events_used - {p}), hosts_ips | {ev[p]["host_ip"]},
                                        hosts_ids | {ev[p]["host_id"]})
                    if not deeper:
                        candidates.append(path_segments)
                    any_verified = True
                else:
                    candidates.append(path_segments)            # terminal: ambiguous / rejected candidates are not extended
        return any_verified

    for e in entries:
        origin_ip = ev[e]["src_ip"]                             # where the entry connection came from
        origin_ids = {o["host_id"] for o in ev.values() if o["kind"] == "host" and o["host_ip"] == origin_ip}
        extend(e, [], set(), {ev[e]["host_ip"], origin_ip}, {ev[e]["host_id"]} | origin_ids)

    paths = _drop_subpaths([_path_dict(c, rec, ev) for c in candidates])
    order = {"supported": 0, "ambiguous": 1, "rejected": 2}
    paths.sort(key=lambda p: (order[p["status"]], 0 if p["qualifies"] else 1, -p["n_hops"], p["events"][0]["ts_s"],
                              tuple(e["local_index"] for e in p["events"])))
    sup = [p for p in paths if p["status"] == "supported"]
    qual = [p for p in sup if p["qualifies"]]
    svc = [p for p in sup if not p["qualifies"]]
    amb = [p for p in paths if p["status"] == "ambiguous"]
    amb_q = [p for p in amb if p["qualifies"]]
    rej = [p for p in paths if p["status"] == "rejected"]
    reasoning.append(f"{n} events on {n_hosts} host(s); relationships observed: {rel_summary}; verified host-to-host hops: {n_hops_total}")
    if n_hops_total and not entries:
        limitations.append("host-to-host hops exist but no inbound accept event is present to anchor a progression")
    incoming = {b_ for lst in internal.values() for (b_, _r, _d) in lst}
    if hops:
        anchored = {ch[-1][1] for en in entries for ch in pivot_chains(en)}
        unanchored = sorted(ev[c]["node"] for c in hops if c not in anchored)
        if unanchored:
            limitations.append("host-to-host connection(s) without observed inbound/process context in this scenario: "
                               + ", ".join(unanchored[:5]))
    if problems:
        limitations.append(f"{len(problems)} link(s) failed re-verification and were excluded from traversal")
    limitations.append("a legitimate service/proxy chain has the same structure (Stage 3b audit); intent is not established")
    if qual:
        limitations.append("a process-lineage path can still be ordinary administration (for example SSH hopping)")

    if qual:
        best = qual[0]
        types = sorted({t_ for p in sup for t_ in p["evidence_types"]})
        reasoning.append(f"{len(qual)} verified path(s) with a process-lineage pivot; best spans {best['n_hops']} cross-host "
                         f"hop(s) over {len(best['hosts'])} hosts: " + " -> ".join(h["host_id"] for h in best["hosts"]))
        reasoning.append("pivot evidence types present in this scenario: " + ", ".join(types))
        reasoning.append("each hop's direction comes from the connect/accept event fields; each pivot step was re-verified "
                         "against process fields and has strictly increasing timestamps")
        if svc:
            limitations.append(f"{len(svc)} additional verified path(s) rely on service hand-off only (informational)")
        if amb or rej:
            limitations.append(f"{len(amb)} ambiguous and {len(rej)} rejected alternative candidate(s) also recorded")
        return done(SUPPORTED, "verified_cross_host_progression", paths, _involved(qual), rel_summary)
    if search["truncated"]:
        reasoning.append("the candidate search reached its bound, so absence of a path cannot be concluded")
        return done(INSUFFICIENT, "search_bound_reached", paths, rel=rel_summary)
    if amb_q:
        reasoning.append(f"{len(amb_q)} structurally complete candidate(s) with a process-lineage pivot could not be verified "
                         "because event order is unresolved (equal timestamps between different events)")
        return done(INSUFFICIENT, "ambiguous_event_order", paths, rel=rel_summary)
    if svc:
        reasoning.append(f"{len(svc)} verified cross-host chain(s) whose pivots are all service hand-offs (the same long-lived "
                         "service process accepts, then connects onward): " + " -> ".join(h["host_id"] for h in svc[0]["hosts"]))
        reasoning.append("no process-spawn / lineage evidence accompanies the hand-off, so this is ordinary application-service "
                         "progression and is reported for information only")
        limitations.append(ASSESSMENT_MEANING[SERVICE_CHAIN_ONLY])
        if amb or rej:
            limitations.append(f"{len(amb)} ambiguous and {len(rej)} rejected alternative candidate(s) also recorded")
        return done(SERVICE_CHAIN_ONLY, "verified_service_chain_without_process_lineage", paths, _involved(svc), rel_summary)
    if amb:
        reasoning.append(f"{len(amb)} structurally complete candidate(s) could not be verified because event order is unresolved "
                         "(equal timestamps between different events)")
        return done(INSUFFICIENT, "ambiguous_event_order", paths, rel=rel_summary)
    if problems:
        reasoning.append("link evidence contradicted the original event fields; the scenario cannot be assessed reliably")
        return done(INSUFFICIENT, "evidence_inconsistent", paths, rel=rel_summary)
    limitations.append(ASSESSMENT_MEANING[NO_PATH])
    if rej:
        reasoning.append(f"{len(rej)} candidate(s) were rejected: contradicted by chronology or direction evidence")
        return done(NO_PATH, "candidates_contradicted_by_chronology", paths, rel=rel_summary)
    if search["origin_revisits_rejected"]:
        reasoning.append("the only onward route leads back to the host the entry connection came from; returning to the origin "
                         "is not movement to a new host")
        return done(NO_PATH, "only_route_returns_to_origin", paths, rel=rel_summary)
    if n_hops_total:
        reasoning.append("host-to-host communication is present, but none follows an inbound event through a verified process "
                         "pivot on the source host; an ordinary connection alone is not movement evidence")
        flags = []
        dest = sorted(ev[c]["node"] for c, lst in hops.items() for (p_, _r, _d) in lst if p_ in internal and c not in incoming)
        if dest:
            flags.append("destination_side_activity_incomplete_predecessor")
            limitations.append("process activity continues on the accepting host (" + ", ".join(dest[:3]) + " -> ...) but the "
                               "connecting event has no predecessor in this scenario; the earlier part of a chain may lie "
                               "outside the scenario. Not treated as a positive finding and nothing is inferred")
        return done(NO_PATH, "host_to_host_without_pivot_context", paths, rel=rel_summary, flags=flags)
    reasoning.append("no directed host-to-host communication was observed in the scenario" +
                     ("" if n_hosts > 1 else " (all events are on a single host)"))
    return done(NO_PATH, "no_cross_host_communication", paths, rel=rel_summary)


def _entry_context(entry, ev, iot_entry, rec):
    e = ev[entry]
    if entry in iot_entry:
        a, row, detail = iot_entry[entry][0]
        return {"source_class": "iot_flow_matched", "iot_event": ev[a]["node"], "iot_device": ev[a]["iot_device_id"],
                "match": detail, "evidence_ref": _ref(rec, row), "source_ip": e["src_ip"]}
    if any(o["kind"] == "host" and o["host_ip"] == e["src_ip"] for o in ev.values()):
        return {"source_class": "host_in_scenario", "source_ip": e["src_ip"]}
    return {"source_class": "unclassified_source", "source_ip": e["src_ip"]}


def _path_dict(segments, rec, ev):
    status = "supported" if all(s["status"] == "verified" for s in segments) else         ("rejected" if any(s["status"] == "rejected" for s in segments) else "ambiguous")
    by_node = {e["node"]: e for e in ev.values()}
    events, seen = [], set()

    def add(view):
        if view["event"] not in seen:
            seen.add(view["event"])
            events.append(view)

    for s in segments:
        add(s["entry"])
        for st in s["pivot_links"]:
            add(_event_view(by_node[st["to_event"]]))
        add(s["remote_accept"])
    hosts, hs = [], set()
    for v in events:
        if v["kind"] == "host" and v["host_ip"] not in hs:
            hs.add(v["host_ip"])
            hosts.append({"host_id": v["host_id"], "host_ip": v["host_ip"]})
    kinds = [s["pivot_kind"] for s in segments]
    evidence_types = sorted({t_ for k in kinds for t_ in (("process_lineage",) if k == "process_lineage" else
                                                          ("service_handoff",) if k == "service_handoff" else
                                                          ("process_lineage", "service_handoff"))})
    qualifies = "process_lineage" in evidence_types
    digest = hashlib.sha1("|".join([str(rec.scenario_id)] + [e["event"] for e in events]).encode()).hexdigest()[:10]
    classification = status if status != "supported" else ("process_lineage_progression" if qualifies else "service_chain_only")
    return {"path_id": f"F2-{rec.scenario_id}-{digest}", "status": status, "qualifies": qualifies,
            "classification": classification, "evidence_types": evidence_types, "n_hops": len(segments),
            "pivot_kinds": kinds, "hosts": hosts, "events": events, "segments": segments,
            "reasons": [r for s in segments for r in s["reasons"]]}


def _drop_subpaths(paths):
    """Remove a path whose event sequence is a contiguous part of a longer path with the same status (for example the
    B -> C part of A -> B -> C, which is found again when the search starts at B's accept)."""
    keys = [tuple(e["event"] for e in p["events"]) for p in paths]

    def inside(small, big):
        m = len(small)
        return len(big) > m and any(big[i:i + m] == small for i in range(len(big) - m + 1))

    return [p for i, p in enumerate(paths)
            if not any(j != i and paths[j]["status"] == p["status"] and inside(keys[i], keys[j]) for j in range(len(paths)))]


def _involved(supported_paths):
    hosts, procs, events = {}, {}, {}
    for p in supported_paths:
        for v in p["events"]:
            events[v["event"]] = {"timestamp": v["timestamp"], "kind": v["kind"]}
            if v["kind"] == "host":
                hosts[v["host_ip"]] = v["host_id"]
                procs[(v["host_id"], v["process_id"])] = v["process_name"]
    return {"hosts": [{"host_ip": k, "host_id": v} for k, v in sorted(hosts.items())],
            "processes": [{"host_id": h, "process_id": pid, "process_name": nm} for (h, pid), nm in sorted(procs.items())],
            "events": dict(sorted(events.items()))}


def check_f1_gate(handoff: F1F2Handoff):
    """Cross-check every handoff record against the authoritative F1 results (`handoff.f1_scores`). Returns a list of
    violations (empty = consistent). Nothing is recomputed or modified: scores and decisions are only compared."""
    problems = []
    fs = handoff.f1_scores
    if not fs["scenario_id"].is_unique:
        return ["authoritative F1 results contain duplicate scenario ids"]
    auth = fs.set_index("scenario_id")
    for sid, rec in handoff.records.items():
        if rec.scenario_id != sid:
            problems.append(f"scenario {sid}: record carries scenario id {rec.scenario_id}")
        if sid not in auth.index:
            problems.append(f"scenario {sid}: not present in the authoritative F1 results")
            continue
        row = auth.loc[sid]
        a_score, a_thr = float(row["anomaly_score"]), float(row["threshold"])
        if not (bool(row["is_suspicious"]) and row["prediction"] == "Suspicious" and a_score > a_thr):
            problems.append(f"scenario {sid}: F1 did not classify it Suspicious (prediction={row['prediction']}, "
                            f"score={a_score:.6f}, threshold={a_thr:.6f})")
        if a_thr != float(handoff.threshold):
            problems.append(f"scenario {sid}: F1 threshold {a_thr} differs from the handoff's locked threshold {handoff.threshold}")
        f = rec.f1
        if f.get("prediction") != row["prediction"]:
            problems.append(f"scenario {sid}: record prediction {f.get('prediction')!r} != F1 result {row['prediction']!r}")
        if f.get("anomaly_score") != a_score:
            problems.append(f"scenario {sid}: record score {f.get('anomaly_score')!r} != F1 score {a_score!r}")
        if f.get("threshold") != a_thr:
            problems.append(f"scenario {sid}: record threshold {f.get('threshold')!r} != F1 threshold {a_thr!r}")
    return problems


def analyse_handoff(handoff: F1F2Handoff, max_candidates=MAX_CANDIDATES) -> F2Report:
    """Analyse every record of an `F1F2Handoff`. The records are first cross-checked against `handoff.f1_scores`; a
    contradiction raises `F1GateError` before any analysis. A scenario that raises during analysis is reported in
    `errors`; nothing is silently dropped."""
    violations = check_f1_gate(handoff)
    if violations:
        raise F1GateError(violations)
    out, errors = {}, {}
    for sid, rec in handoff.records.items():
        try:
            out[sid] = analyse_scenario(rec, max_candidates)
        except NotSuspiciousError:
            raise
        except Exception as exc:            # reported, never swallowed
            errors[sid] = f"{type(exc).__name__}: {exc}"
    return F2Report(out, errors)
