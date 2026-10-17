"""Tests for Function 2 (xdr_c3.lateral_movement).

Part 1 uses small, hand-specified scenarios with known valid and invalid paths. They test ALGORITHM correctness only;
they are not findings. Part 2 runs Function 2 on the real handoff records."""
import ast
import copy
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xdr_c3 import f1_f2_handoff as h
from xdr_c3 import lateral_movement as lm
from xdr_c3 import stage2_validate as s2

F1 = {"anomaly_score": 4.0, "threshold": 2.5825197985552437, "prediction": "Suspicious", "method": "knn_embedding",
      "k": 5, "model": "R-GCN-GAE", "checkpoint_epoch": 40}
HOSTS = {"A": ("web-edge-01", "10.20.1.11"), "B": ("api-node-01", "10.20.1.21"), "C": ("db-core-01", "10.20.2.11")}
T0 = 1546300800            # 01/01/2019 00:00:00 UTC


def stamp(ts_s):
    return pd.Timestamp(ts_s, unit="s").strftime("%d/%m/%Y %H:%M:%S")


# --------------------------------------------------------------------------- scenario builder (fixtures)
class Scn:
    """Builds a ScenarioHandoff record from explicit events and links (original link orientation)."""

    def __init__(self, f1=None):
        self.events, self.links, self.f1 = [], [], dict(f1 or F1)

    def host(self, h_, t, etype, pid, ppid=1, name="proc", src="-", dst="-", sport="", dport="", proto="-", **kw):
        hid, hip = HOSTS[h_]
        self.events.append(dict(kind="host", t=t, host_id=hid, host_ip=hip, process_id=pid, parent_process_id=ppid,
                                process_name=name, event_type=etype, src_ip=src, dst_ip=dst, src_port=sport,
                                dst_port=dport, protocol=proto, **kw))
        return len(self.events) - 1

    def accept(self, h_, t, pid, src, sport, dport, **kw):
        return self.host(h_, t, "network_accept", pid, 1, "sshd", src, HOSTS[h_][1], sport, dport, "TCP", **kw)

    def connect(self, h_, t, pid, dst_host, sport, dport, **kw):
        return self.host(h_, t, "network_connect", pid, kw.pop("ppid", 1), kw.pop("name", "ssh"), HOSTS[h_][1],
                         HOSTS[dst_host][1], sport, dport, "TCP", **kw)

    def pstart(self, h_, t, pid, ppid, name="bash"):
        return self.host(h_, t, "process_start", pid, ppid, name)

    def iot(self, t, src, dst, sport, dport, device="plc-1"):
        self.events.append(dict(kind="iot", t=t, iot_device_id=device, src_ip=src, dst_ip=dst, src_port=sport,
                                dst_port=dport, protocol="TCP", service="SSH"))
        return len(self.events) - 1

    def link(self, a, b, lt, points=2, gap=1.0):
        self.links.append((a, b, lt, points, gap))

    def hop(self, c, p):
        self.link(c, p, "connect_accept", 3, 0.0)

    def record(self, scenario_id=1, integrity=None):
        ev = self.events
        n = len(ev)
        host_rows, iot_rows, nodes = [], [], []
        for i, e in enumerate(ev):
            ts = stamp(T0 + e["t"])
            nodes.append(dict(local_index=i, node=f"{e['kind']}:{i}", kind=e["kind"], row=i, timestamp=ts, ts_s=T0 + e["t"],
                              host_ip=e.get("host_ip", ""), model_node_error=0.01))
            rec = {k: v for k, v in e.items() if k not in ("kind", "t")}
            rec.update(local_index=i, timestamp=ts, row=i)
            (host_rows if e["kind"] == "host" else iot_rows).append(rec)
        nodes = pd.DataFrame(nodes)
        name = lambda i: nodes.at[i, "node"]
        ev_rows = []
        for k, (a, b, lt, pts, gap) in enumerate(self.links):
            ev_rows.append(dict(scenario_id=scenario_id, edge_index=k, local_a=min(a, b), local_b=max(a, b), node_a=name(a),
                                node_b=name(b), link_type=lt, points=pts, gap_sec=gap))
        evidence = pd.DataFrame(ev_rows, columns=["scenario_id", "edge_index", "local_a", "local_b", "node_a", "node_b",
                                                  "link_type", "points", "gap_sec"])
        edges = evidence[["edge_index", "local_a", "local_b"]].assign(model_edge_error_mean=0.01)
        hcols = ["local_index", "timestamp", "host_id", "host_ip", "process_id", "parent_process_id", "process_name",
                 "event_type", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "threat_state", "row"]
        icols = ["local_index", "timestamp", "iot_device_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
                 "service", "row"]
        he = pd.DataFrame(host_rows)
        for c in hcols:
            if c not in he:
                he[c] = "normal" if c == "threat_state" else ""
        ie = pd.DataFrame(iot_rows, columns=icols) if iot_rows else pd.DataFrame(columns=icols)
        return h.ScenarioHandoff(
            scenario_id=scenario_id, f1=dict(self.f1), model_signals={"graph_errors": {}}, nodes=nodes,
            host_events=he[hcols], iot_events=ie, edges=edges, evidence=evidence, observed={},
            component_columns={}, integrity_problems=list(integrity or []))


def chain(s, t_accept=100, t_child=110, t_connect=120, t_remote=120, dst="B", **kw):
    """entry accept on A -> process_start child -> connect (same child pid) -> accept on B.  Returns the event indices."""
    acc = s.accept("A", t_accept, 500, "10.30.1.5", 51000, 22)
    ps = s.pstart("A", t_child, 900, 500)
    con = s.connect("A", t_connect, 900, dst, 40000, 22, **kw)
    rem = s.accept(dst, t_remote, 700, HOSTS["A"][1], 40000, 22)
    s.link(acc, ps, "entry_to_child", 3, t_child - t_accept)
    s.link(ps, con, "same_process", 2, t_connect - t_child)
    s.hop(con, rem)
    return acc, ps, con, rem


def valid_scenario():
    s = Scn()
    ids = chain(s)
    return s, ids


# =========================================================================== F1 gating
def test_only_f1_suspicious_scenarios_are_accepted():
    s, _ = valid_scenario()
    normal = s.record()
    normal.f1.update(prediction="Normal", anomaly_score=1.0)
    with pytest.raises(lm.NotSuspiciousError):
        lm.analyse_scenario(normal)
    inconsistent = s.record()
    inconsistent.f1.update(anomaly_score=1.0)                      # says Suspicious but score is below the threshold
    with pytest.raises(lm.NotSuspiciousError):
        lm.analyse_scenario(inconsistent)
    ok = lm.analyse_scenario(s.record())
    assert ok.f1 == F1 and ok.f1 is not s.f1                       # F1 result carried unchanged (a copy)


# =========================================================================== genuine progressions
def test_verified_cross_host_progression_is_supported():
    s, (acc, ps, con, rem) = valid_scenario()
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.SUPPORTED and a.reason_code == "verified_cross_host_progression"
    (path,) = a.supported_paths
    assert [e["event"] for e in path["events"]] == [f"host:{acc}", f"host:{ps}", f"host:{con}", f"host:{rem}"]
    assert path["n_hops"] == 1 and [x["host_id"] for x in path["hosts"]] == [HOSTS["A"][0], HOSTS["B"][0]]
    seg = path["segments"][0]
    assert seg["hop"]["source"]["host_ip"] == HOSTS["A"][1] and seg["hop"]["destination"]["host_ip"] == HOSTS["B"][1]
    assert seg["hop"]["connect_event"] == f"host:{con}" and seg["hop"]["accept_event"] == f"host:{rem}"
    assert seg["pivot_kind"] == "process_lineage" and [x["order"] for x in seg["pivot_links"]] == ["ordered", "ordered"]
    assert a.involved["hosts"] and a.involved["processes"] and set(a.involved["events"]) == {e["event"] for e in path["events"]}
    assert "potential" in a.evidence_policy["claim"] and "intent" in a.evidence_policy["claim"]


def test_multi_hop_path_reports_only_the_maximal_verified_path():
    s = Scn()
    acc, ps, con, rem = chain(s)                                  # A -> B
    con2 = s.connect("B", 140, 700, "C", 41000, 22)               # B's accepting service connects onward
    rem2 = s.accept("C", 141, 800, HOSTS["B"][1], 41000, 22)
    s.link(rem, con2, "service_handoff", 2, 20)
    s.hop(con2, rem2)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.SUPPORTED
    (path,) = a.supported_paths                                   # the A->B prefix is not reported separately
    assert path["n_hops"] == 2 and path["pivot_kinds"] == ["process_lineage", "service_handoff"]
    assert [x["host_id"] for x in path["hosts"]] == [HOSTS["A"][0], HOSTS["B"][0], HOSTS["C"][0]]
    assert [e["event"] for e in path["events"]] == [f"host:{i}" for i in (acc, ps, con, rem, con2, rem2)]


def test_same_second_connect_and_accept_is_accepted_because_direction_comes_from_fields():
    s = Scn()
    chain(s, t_connect=120, t_remote=120)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.SUPPORTED
    assert a.supported_paths[0]["segments"][0]["hop"]["accept_minus_connect_s"] == 0
    assert "same_second" in a.supported_paths[0]["segments"][0]["hop"]["time_order"]


def test_iot_entry_context_is_recorded_but_not_required():
    s = Scn()
    iot = s.iot(95, "10.30.1.5", HOSTS["A"][1], 51000, 22)
    acc, ps, con, rem = chain(s)
    s.link(iot, acc, "iot_host_match", 3, 5)
    a = lm.analyse_scenario(s.record())
    assert a.supported_paths[0]["segments"][0]["entry_context"]["source_class"] == "iot_flow_matched"
    assert a.supported_paths[0]["segments"][0]["entry_context"]["iot_device"] == "plc-1"
    s2_, _ = valid_scenario()
    assert lm.analyse_scenario(s2_.record()).supported_paths[0]["segments"][0]["entry_context"]["source_class"] == "unclassified_source"


# =========================================================================== not enough evidence for movement
def test_host_to_host_connection_alone_is_not_movement():
    s = Scn()
    ps = s.pstart("A", 100, 900, 1)
    con = s.connect("A", 110, 900, "B", 40000, 22)
    rem = s.accept("B", 110, 700, HOSTS["A"][1], 40000, 22)
    s.link(ps, con, "same_process", 2, 10)
    s.hop(con, rem)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.NO_PATH and a.reason_code == "host_to_host_without_pivot_context"
    assert any("without observed inbound" in x for x in a.limitations)


def test_single_host_anomalous_scenario_has_no_supporting_path():
    s = Scn()
    acc = s.accept("A", 100, 500, "10.30.1.5", 51000, 22)
    ps = s.pstart("A", 110, 900, 500)
    ps2 = s.pstart("A", 120, 901, 900)
    s.link(acc, ps, "entry_to_child", 3, 10)
    s.link(ps, ps2, "parent_child", 2, 10)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.NO_PATH and a.reason_code == "no_cross_host_communication" and not a.paths
    assert any("single host" in x for x in a.reasoning)


def test_two_event_scenario_is_insufficient_even_with_a_host_to_host_link():
    s = Scn()
    con = s.connect("A", 100, 900, "B", 40000, 22)
    rem = s.accept("B", 100, 700, HOSTS["A"][1], 40000, 22)
    s.hop(con, rem)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.INSUFFICIENT and a.reason_code == "fewer_than_3_events" and a.supported_paths == []


def test_context_links_across_hosts_do_not_create_a_path():
    """Graph paths over non-movement relations (IoT match, same remote source, flow links) are not communication paths."""
    s = Scn()
    iot = s.iot(90, "10.30.1.5", HOSTS["A"][1], 51000, 22)
    a1 = s.accept("A", 100, 500, "10.30.1.5", 51000, 22)
    a2 = s.accept("A", 130, 501, "10.30.1.5", 51001, 22)
    b1 = s.accept("B", 140, 700, "10.30.1.5", 51002, 22)
    s.link(iot, a1, "iot_host_match", 3, 10)
    s.link(a1, a2, "same_remote_source", 2, 30)
    s.link(a2, b1, "same_remote_source", 2, 10)          # (as a bridge in the graph; not host-internal in reality)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.NO_PATH and a.paths == []


# =========================================================================== chronology, direction, ambiguity
def test_chronologically_inconsistent_chain_is_rejected_not_supported():
    s = Scn()
    chain(s, t_accept=100, t_child=90, t_connect=120)             # child process "starts" before the accept
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.NO_PATH and a.reason_code == "candidates_contradicted_by_chronology"
    assert a.paths and a.paths[0]["status"] == "rejected" and a.supported_paths == []
    assert any("earlier timestamp" in r for r in a.paths[0]["reasons"])


def test_accept_before_connect_is_rejected():
    s = Scn()
    chain(s, t_connect=120, t_remote=119)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.NO_PATH and a.paths[0]["status"] == "rejected"


def test_hop_slower_than_stage3_tolerance_is_rejected():
    s = Scn()
    chain(s, t_connect=120, t_remote=130)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.NO_PATH and a.paths[0]["status"] == "rejected"


def test_equal_timestamps_between_chain_events_stay_ambiguous():
    s = Scn()
    chain(s, t_accept=100, t_child=100, t_connect=120)            # order of accept and child process is unresolved
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.INSUFFICIENT and a.reason_code == "ambiguous_event_order"
    assert a.paths[0]["status"] == "ambiguous" and a.supported_paths == []
    assert any("equal timestamps" in r for r in a.paths[0]["reasons"])


def test_service_handoff_with_equal_timestamps_is_ambiguous():
    s = Scn()
    acc = s.accept("A", 100, 500, "10.30.1.5", 51000, 22)
    con = s.connect("A", 100, 500, "B", 40000, 22, name="nginx")
    rem = s.accept("B", 100, 700, HOSTS["A"][1], 40000, 22)
    s.link(acc, con, "service_handoff", 2, 0)
    s.hop(con, rem)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.INSUFFICIENT and a.reason_code == "ambiguous_event_order"


def test_reversed_hop_orientation_is_not_trusted():
    s = Scn()
    acc = s.accept("A", 100, 500, "10.30.1.5", 51000, 22)
    ps = s.pstart("A", 110, 900, 500)
    con = s.connect("A", 120, 900, "B", 40000, 22)
    rem = s.accept("B", 120, 700, HOSTS["A"][1], 40000, 22)
    s.link(acc, ps, "entry_to_child", 3, 10)
    s.link(ps, con, "same_process", 2, 10)
    s.link(rem, con, "connect_accept", 3, 0)                      # orientation reversed: accept -> connect
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.INSUFFICIENT and a.reason_code == "evidence_inconsistent"
    assert any("orientation" in p for p in a.evidence_problems) and a.supported_paths == []


def test_missing_direction_evidence_is_not_assumed():
    s = Scn()
    acc = s.accept("A", 100, 500, "10.30.1.5", 51000, 22)
    ps = s.pstart("A", 110, 900, 500)
    con = s.host("A", 120, "network_connect", 900, 500, "ssh", "10.99.9.9", HOSTS["B"][1], 40000, 22, "TCP")  # src != host
    rem = s.accept("B", 120, 700, "10.99.9.9", 40000, 22)
    s.link(acc, ps, "entry_to_child", 3, 10)
    s.link(ps, con, "same_process", 2, 10)
    s.hop(con, rem)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.INSUFFICIENT and a.reason_code == "evidence_inconsistent"


def test_ambiguous_network_tuple_is_not_supported():
    s = Scn()
    acc = s.accept("A", 100, 500, "10.30.1.5", 51000, 22)
    ps = s.pstart("A", 110, 900, 500)
    con = s.connect("A", 120, 900, "B", 40000, 22)
    rem = s.accept("B", 120, 700, HOSTS["A"][1], 40001, 22)       # different source port than the connect
    s.link(acc, ps, "entry_to_child", 3, 10)
    s.link(ps, con, "same_process", 2, 10)
    s.hop(con, rem)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.INSUFFICIENT and any("tuples differ" in p for p in a.evidence_problems)


def test_undirected_graph_edge_direction_is_not_used_for_the_hop():
    """Stage 4 stores edges bidirectionally with lo < hi. Put the ACCEPT at the lower index: the reported direction must
    still follow the event fields (connect host -> accept host), not the graph's index order."""
    s = Scn()
    rem = s.accept("B", 120, 700, HOSTS["A"][1], 40000, 22)       # local index 0 (lower than the connect)
    acc = s.accept("A", 100, 500, "10.30.1.5", 51000, 22)
    ps = s.pstart("A", 110, 900, 500)
    con = s.connect("A", 120, 900, "B", 40000, 22)
    s.link(acc, ps, "entry_to_child", 3, 10)
    s.link(ps, con, "same_process", 2, 10)
    s.hop(con, rem)
    rec = s.record()
    row = rec.evidence[rec.evidence.link_type == "connect_accept"].iloc[0]
    assert row.local_a < row.local_b and row.node_a == f"host:{con}" and row.local_a == rem   # graph order != direction
    a = lm.analyse_scenario(rec)
    hop = a.supported_paths[0]["segments"][0]["hop"]
    assert hop["source"]["host_id"] == HOSTS["A"][0] and hop["destination"]["host_id"] == HOSTS["B"][0]


# =========================================================================== cycles, duplicates, bounds
def test_host_cycle_is_not_followed():
    s = Scn()
    acc, ps, con, rem = chain(s)                                  # A -> B
    con2 = s.connect("B", 140, 700, "A", 42000, 22, name="sshd")  # B -> back to A
    rem2 = s.accept("A", 141, 502, HOSTS["B"][1], 42000, 22)
    s.link(rem, con2, "service_handoff", 2, 20)
    s.hop(con2, rem2)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.SUPPORTED
    assert all(p["n_hops"] == 1 for p in a.paths)                 # the return hop to host A is never added
    assert len({e["event"] for e in a.supported_paths[0]["events"]}) == len(a.supported_paths[0]["events"])


def test_cycle_in_internal_links_terminates():
    s = Scn()
    acc, ps, con, rem = chain(s)
    s.link(con, ps, "same_process", 2, 10)                        # internal cycle ps -> con -> ps
    s.link(ps, acc, "parent_child", 2, 10)                        # and back to the entry
    a = lm.analyse_scenario(s.record())
    assert a.assessment in (lm.SUPPORTED, lm.INSUFFICIENT, lm.NO_PATH)
    for p in a.paths:
        evs = [e["event"] for e in p["events"]]
        assert len(evs) == len(set(evs))


def test_duplicate_relationships_are_collapsed_and_reported():
    s, _ = valid_scenario()
    rec = s.record()
    dup = dataclasses.replace(rec, evidence=pd.concat([rec.evidence, rec.evidence.iloc[[0]]], ignore_index=True))
    a, b = lm.analyse_scenario(rec), lm.analyse_scenario(dup)
    assert b.assessment == a.assessment and b.supported_paths == a.supported_paths
    assert any("duplicate relationship" in x for x in b.limitations)


def test_search_bound_is_reported_not_hidden():
    s = Scn()
    for _ in range(3):
        chain(s)
    a = lm.analyse_scenario(s.record(), max_candidates=1)
    assert a.search["truncated"] is True
    if a.assessment != lm.SUPPORTED:
        assert a.assessment == lm.INSUFFICIENT and a.reason_code == "search_bound_reached"


def test_two_independent_progressions_are_ordered_deterministically():
    s = Scn()
    chain(s, t_accept=100, t_child=110, t_connect=120, t_remote=120)
    chain(s, t_accept=200, t_child=210, t_connect=220, t_remote=221)
    rec = s.record()
    a = lm.analyse_scenario(rec)
    assert len(a.supported_paths) == 2
    assert [p["events"][0]["ts_s"] for p in a.supported_paths] == sorted(p["events"][0]["ts_s"] for p in a.supported_paths)
    assert a.to_dict() == lm.analyse_scenario(rec).to_dict()                           # deterministic
    shuffled = dataclasses.replace(rec, evidence=rec.evidence.sample(frac=1, random_state=2).reset_index(drop=True))
    assert lm.analyse_scenario(shuffled).to_dict() == a.to_dict()                      # independent of row order


# =========================================================================== corrupted / inconsistent evidence
def test_failed_integrity_record_is_insufficient():
    s, _ = valid_scenario()
    a = lm.analyse_scenario(s.record(integrity=["edge 0 has no relationship evidence"]))
    assert a.assessment == lm.INSUFFICIENT and a.reason_code == "handoff_integrity_problem" and a.evidence_problems


def test_link_contradicting_event_fields_is_excluded_and_reported():
    s = Scn()
    acc = s.accept("A", 100, 500, "10.30.1.5", 51000, 22)
    ps = s.pstart("A", 110, 900, 777)                              # parent pid does not match the accepting process
    con = s.connect("A", 120, 900, "B", 40000, 22)
    rem = s.accept("B", 120, 700, HOSTS["A"][1], 40000, 22)
    s.link(acc, ps, "entry_to_child", 3, 10)                       # claims acc is the parent, but ppid says 777
    s.link(ps, con, "same_process", 2, 10)
    s.hop(con, rem)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.INSUFFICIENT and a.reason_code == "evidence_inconsistent"
    assert any("entry_to_child" in p and "failed re-verification" in p for p in a.evidence_problems)


def test_link_to_event_outside_the_scenario_is_reported():
    s, _ = valid_scenario()
    rec = s.record()
    ev = rec.evidence.copy()
    ev.loc[0, "node_b"] = "host:99999"
    a = lm.analyse_scenario(dataclasses.replace(rec, evidence=ev))
    assert any("outside the scenario" in p for p in a.evidence_problems) and a.assessment != lm.SUPPORTED


def test_cross_host_link_between_same_host_events_is_not_a_hop():
    s = Scn()
    acc = s.accept("A", 100, 500, "10.30.1.5", 51000, 22)
    ps = s.pstart("A", 110, 900, 500)
    con = s.connect("A", 120, 900, "A", 40000, 22)                 # connect to itself
    rem = s.accept("A", 120, 700, HOSTS["A"][1], 40000, 22)
    s.link(acc, ps, "entry_to_child", 3, 10)
    s.link(ps, con, "same_process", 2, 10)
    s.hop(con, rem)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.INSUFFICIENT and any("same host" in p for p in a.evidence_problems)


# =========================================================================== independence from model / labels
def test_decision_ignores_scores_errors_component_predictions_and_labels():
    s, _ = valid_scenario()
    base = lm.analyse_scenario(s.record())
    rec = s.record()
    rec2 = dataclasses.replace(
        rec, nodes=rec.nodes.assign(model_node_error=99.0), edges=rec.edges.assign(model_edge_error_mean=99.0),
        host_events=rec.host_events.assign(threat_state="suspicious"), model_signals={"graph_errors": {"recon_total": 1e9}},
        f1=dict(rec.f1, anomaly_score=99.0))
    other = lm.analyse_scenario(rec2)
    assert (other.assessment, other.reason_code) == (base.assessment, base.reason_code)
    assert other.paths == base.paths and other.involved == base.involved
    assert "component predictions (C1/C2)" in base.evidence_policy["not_used_for_decision"]
    # negative direction: strongly "normal-looking" errors cannot remove a verified path either
    rec3 = dataclasses.replace(rec, nodes=rec.nodes.assign(model_node_error=0.0))
    assert lm.analyse_scenario(rec3).assessment == lm.SUPPORTED


def test_f2_code_has_no_ground_truth_or_model_error_inputs():
    tree = ast.parse(Path(lm.__file__).read_text(encoding="utf-8"))
    banned = {"label", "type", "attack_type", "node_label", "is_attack", "model_node_error", "model_edge_error_forward",
              "model_edge_error_reverse", "model_edge_error_mean", "threat_state", "threat_confidence_score",
              "trace_mode", "predicted_class", "confidence_score", "graph_errors"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in banned, node.value
    gating_only = {"anomaly_score", "threshold", "prediction"}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name not in ("analyse_scenario", "check_f1_gate")]:
        used = {c.value for c in ast.walk(fn) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
        assert not used & gating_only, fn.name


# =========================================================================== traceability and no fabrication
def test_every_path_reference_resolves_to_original_events_and_evidence():
    s = Scn()
    iot = s.iot(95, "10.30.1.5", HOSTS["A"][1], 51000, 22)
    acc, ps, con, rem = chain(s)
    s.link(iot, acc, "iot_host_match", 3, 5)
    rec = s.record()
    a = lm.analyse_scenario(rec)
    nodes = dict(zip(rec.nodes["node"], rec.nodes["timestamp"]))
    ev_rows = {(r.node_a, r.node_b, r.link_type, r.points, r.gap_sec, r.edge_index) for r in rec.evidence.itertuples()}
    for p in a.paths:
        for e in p["events"]:
            assert e["event"] in nodes and e["timestamp"] == nodes[e["event"]]
        for seg in p["segments"]:
            refs = [st["evidence_ref"] for st in seg["pivot_links"]] + [seg["hop"]["evidence_ref"]]
            for r in refs:
                assert (r["node_a"], r["node_b"], r["link_type"], r["points"], r["gap_sec"], r["edge_index"]) in ev_rows
            assert seg["hop"]["connect_event"] == seg["hop"]["evidence_ref"]["node_a"]
            assert seg["hop"]["accept_event"] == seg["hop"]["evidence_ref"]["node_b"]


def test_no_fabricated_conclusions_for_non_qualifying_inputs():
    for build in ("single_host", "unanchored_hop", "two_events"):
        s = Scn()
        if build == "single_host":
            acc = s.accept("A", 100, 500, "10.30.1.5", 51000, 22); ps = s.pstart("A", 110, 900, 500)
            s.link(acc, ps, "entry_to_child", 3, 10)
        elif build == "unanchored_hop":
            con = s.connect("A", 110, 900, "B", 40000, 22); rem = s.accept("B", 110, 700, HOSTS["A"][1], 40000, 22)
            ps = s.pstart("A", 100, 900, 1); s.link(ps, con, "same_process", 2, 10); s.hop(con, rem)
        else:
            con = s.connect("A", 110, 900, "B", 40000, 22); rem = s.accept("B", 110, 700, HOSTS["A"][1], 40000, 22); s.hop(con, rem)
        a = lm.analyse_scenario(s.record())
        assert a.assessment != lm.SUPPORTED and a.supported_paths == [] and a.involved == {}


def test_assessment_serialises_with_reasoning_and_f1_context():
    s, _ = valid_scenario()
    d = lm.analyse_scenario(s.record()).to_dict()
    import json
    d2 = json.loads(json.dumps(d))
    assert d2["f1"]["anomaly_score"] == 4.0 and d2["assessment"] == lm.SUPPORTED and d2["reasoning"] and d2["limitations"]


# =========================================================================== real data: all 1,499 suspicious scenarios
@pytest.fixture(scope="session")
def f2(handoff):
    return lm.analyse_handoff(handoff)


def test_real_data_counts_reconcile(f2, handoff):
    c = f2.counts()
    assert c["analysed"] == len(handoff.records) == 1499 and c["processing_errors"] == 0 and f2.errors == {}
    assert c[lm.SUPPORTED] + c[lm.SERVICE_CHAIN_ONLY] + c[lm.NO_PATH] + c[lm.INSUFFICIENT] + c["processing_errors"] == c["analysed"]
    assert set(f2.assessments) == set(handoff.records)
    assert not set(f2.assessments) & set(handoff.normal["scenario_id"])          # Normal scenarios never analysed
    assert (c[lm.SUPPORTED], c[lm.SERVICE_CHAIN_ONLY], c[lm.NO_PATH], c[lm.INSUFFICIENT]) == (141, 23, 1227, 108)


def test_real_f1_context_is_carried_unchanged(f2, handoff):
    for sid, a in list(f2.assessments.items())[::40]:
        assert a.f1 == handoff.records[sid].f1 and a.f1["prediction"] == "Suspicious"


def test_real_supported_paths_are_independently_valid(f2, s2_data):
    host, _ = s2_data.to_stage3_frames()
    H = host.set_index("row")
    checked = 0
    for sid, a in f2.assessments.items():
        if a.assessment not in (lm.SUPPORTED, lm.SERVICE_CHAIN_ONLY):
            assert a.supported_paths == [] and a.involved == {}
            continue
        if a.assessment == lm.SUPPORTED:
            assert a.qualifying_paths and any(
                st["link_type"] in lm.LINEAGE_LINKS for p in a.qualifying_paths for s in p["segments"] for st in s["pivot_links"])
        else:                                                      # informational: only service hand-off pivots
            assert not a.qualifying_paths and all(
                st["link_type"] == "service_handoff" for p in a.supported_paths for s in p["segments"] for st in s["pivot_links"])
        for p in a.supported_paths:
            assert p["status"] == "supported" and len(p["hosts"]) >= 2 and len(p["events"]) >= 3
            assert len({e["event"] for e in p["events"]}) == len(p["events"])
            assert len({x["host_ip"] for x in p["hosts"]}) == len(p["hosts"])
            for seg in p["segments"]:
                row = lambda n: H.loc[int(n.split(":")[1])]
                entry, con, rem = row(seg["entry"]["event"]), row(seg["connect"]["event"]), row(seg["remote_accept"]["event"])
                assert entry.event_type == "network_accept" and con.event_type == "network_connect" and rem.event_type == "network_accept"
                assert entry.host_ip == con.host_ip and con.host_ip != rem.host_ip and con.host_id != rem.host_id
                assert con.src_ip == con.host_ip and con.dst_ip == rem.host_ip == rem.dst_ip
                assert all(con[k] == rem[k] for k in ("src_ip", "dst_ip", "src_port", "dst_port", "protocol"))
                ts = [entry.ts] + [row(st["to_event"]).ts for st in seg["pivot_links"]]
                assert all(b > a_ for a_, b in zip(ts, ts[1:]))                        # strictly increasing chain
                assert 0 <= (rem.ts - con.ts).total_seconds() <= 5
                assert seg["entry"]["timestamp"] == entry.timestamp and seg["remote_accept"]["timestamp"] == rem.timestamp
        checked += 1
    assert checked == 164                                   # 141 potential-movement + 23 informational service chains


def test_real_presence_of_host_to_host_links_does_not_decide_the_outcome(f2, handoff):
    h2h = [sid for sid, r in handoff.records.items() if r.observed["host_to_host_link_present"]]
    kinds = {f2.assessments[s].assessment for s in h2h}
    assert {lm.SUPPORTED, lm.SERVICE_CHAIN_ONLY, lm.NO_PATH, lm.INSUFFICIENT} <= kinds    # presence alone is not decisive
    both = [sid for sid, r in handoff.records.items() if r.observed["host_to_host_link_present"] and r.observed["iot_host_link_present"]]
    assert len(both) == 153
    iot_only = [sid for sid, r in handoff.records.items() if r.observed["iot_host_link_present"] and not r.observed["host_to_host_link_present"]]
    assert all(f2.assessments[s].assessment == lm.NO_PATH for s in iot_only)
    unsupported_h2h = [s for s in h2h if f2.assessments[s].assessment in (lm.NO_PATH, lm.INSUFFICIENT)]
    assert len(h2h) == 317 and len(unsupported_h2h) == 153                                 # 151 no-pivot + 2 two-event


def test_real_two_event_scenarios_are_insufficient(f2, handoff):
    for sid, rec in handoff.records.items():
        if len(rec.nodes) < 3:
            assert f2.assessments[sid].assessment == lm.INSUFFICIENT and f2.assessments[sid].reason_code == "fewer_than_3_events"
    assert sum(len(r.nodes) < 3 for r in handoff.records.values()) == 108


def test_real_results_are_deterministic_and_cap_not_reached(f2, handoff):
    again = lm.analyse_handoff(handoff)
    for sid in list(f2.assessments)[::25]:
        assert again.assessments[sid].to_dict() == f2.assessments[sid].to_dict()
    assert not any(a.search["truncated"] for a in f2.assessments.values())
    assert sum(len(a.evidence_problems) for a in f2.assessments.values()) == 0


def test_real_ground_truth_shuffle_does_not_change_f2(s2_data, s3b, small_result, detector):
    base = h.build_f1_f2_handoff(s2_data, s3b, small_result, detector.contract)
    rng = np.random.default_rng(3)
    shuffled = s2.Stage2Data(s2_data.host, s2_data.iot,
                             s2_data.host_truth.iloc[rng.permutation(len(s2_data.host_truth))].set_axis(s2_data.host_truth.index),
                             s2_data.iot_truth.iloc[rng.permutation(len(s2_data.iot_truth))].set_axis(s2_data.iot_truth.index),
                             s2_data.report)
    other = h.build_f1_f2_handoff(shuffled, s3b, small_result, detector.contract)
    a, b = lm.analyse_handoff(base), lm.analyse_handoff(other)
    assert {k: v.to_dict() for k, v in a.assessments.items()} == {k: v.to_dict() for k, v in b.assessments.items()}


def test_real_normal_handoff_records_are_rejected_if_forced(handoff, result, s2_data, s3b, detector, graphs):
    normal_id = int(handoff.normal["scenario_id"].iloc[0])
    sub = detector.predict_collection(graphs, [normal_id])
    # build a record for a Normal scenario by treating it as flagged, then check F2 refuses it
    forced = dataclasses.replace(sub, scores=sub.scores.assign(is_suspicious=True, prediction="Suspicious"))
    ho = h.build_f1_f2_handoff(s2_data, s3b, forced, detector.contract, strict=False)
    rec = ho.records[normal_id]
    rec.f1["prediction"] = "Normal"
    with pytest.raises(lm.NotSuspiciousError):
        lm.analyse_scenario(rec)



# =========================================================================== refinement: process lineage vs service chain
def handoff_chain(s, hosts=("A", "B"), t0=100, ports=(8080,), pid=500, dport0=40000):
    """entry accept -> service_handoff -> connect -> accept on the next host (repeated for further hosts)."""
    first = s.accept(hosts[0], t0, pid, "203.0.113.7", 51000, 443)
    cur, out = first, [first]
    for i in range(1, len(hosts)):
        con = s.connect(hosts[i - 1], t0 + 50 * i, pid, hosts[i], dport0 + i, ports[min(i - 1, len(ports) - 1)], name="nginx")
        rem = s.accept(hosts[i], t0 + 50 * i + 1, pid + 100 * i, HOSTS[hosts[i - 1]][1], dport0 + i, ports[min(i - 1, len(ports) - 1)])
        s.link(cur, con, "service_handoff", 2, 50)
        s.hop(con, rem)
        out += [con, rem]
        cur = rem
        pid += 100
    return out


def test_process_lineage_only_positive_carries_evidence_type_and_path_id():
    s, _ = valid_scenario()
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.SUPPORTED and a.evidence_types == ["process_lineage"]
    (p,) = a.qualifying_paths
    assert p["classification"] == "process_lineage_progression" and p["qualifies"] and a.service_chain_paths == []
    assert p["path_id"].startswith("F2-1-") and a.to_dict()["qualifying_path_ids"] == [p["path_id"]]
    assert "intent" in a.assessment_meaning and "potential" in a.assessment_meaning


def test_service_handoff_only_single_hop_is_informational_not_potential_movement():
    s = Scn()
    acc, con, rem = handoff_chain(s)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.SERVICE_CHAIN_ONLY and a.reason_code == "verified_service_chain_without_process_lineage"
    assert a.assessment != lm.SUPPORTED and a.qualifying_paths == [] and a.evidence_types == ["service_handoff"]
    (p,) = a.supported_paths
    assert p["classification"] == "service_chain_only" and p["status"] == "supported" and not p["qualifies"]
    assert [e["event"] for e in p["events"]] == [f"host:{acc}", f"host:{con}", f"host:{rem}"]
    assert set(a.involved["events"]) == {e["event"] for e in p["events"]}            # original details preserved
    assert "not potential lateral movement" in a.assessment_meaning
    assert a.to_dict()["service_chain_path_ids"] == [p["path_id"]] and a.to_dict()["qualifying_path_ids"] == []
    assert any("ordinary application-service" in r for r in a.reasoning)


def test_service_chain_only_two_hop_is_one_informational_path():
    s = Scn()
    ev = handoff_chain(s, hosts=("A", "B", "C"), ports=(8080, 5432))
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.SERVICE_CHAIN_ONLY
    (p,) = a.supported_paths                                        # the B -> C suffix is not reported separately
    assert p["n_hops"] == 2 and p["pivot_kinds"] == ["service_handoff", "service_handoff"]
    assert [x["host_id"] for x in p["hosts"]] == [HOSTS[k][0] for k in "ABC"]
    assert [e["event"] for e in p["events"]] == [f"host:{i}" for i in ev]


def test_scenario_with_both_patterns_keeps_the_process_lineage_assessment_and_records_both_types():
    s = Scn()
    handoff_chain(s, hosts=("B", "C"), t0=300)                       # informational service chain B -> C
    chain(s, t_accept=100, t_child=110, t_connect=120, t_remote=120)  # process-lineage progression A -> B
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.SUPPORTED and a.reason_code == "verified_cross_host_progression"
    assert a.evidence_types == ["process_lineage", "service_handoff"]
    assert len(a.qualifying_paths) >= 1 and len(a.service_chain_paths) >= 1
    assert all(p["classification"] == "process_lineage_progression" for p in a.qualifying_paths)
    assert all(p["classification"] == "service_chain_only" for p in a.service_chain_paths)
    ids = [p["path_id"] for p in a.paths]
    assert len(ids) == len(set(ids))
    assert a.paths[0]["qualifies"]                                  # qualifying paths are listed first
    assert any("service hand-off only (informational)" in x for x in a.limitations)
    assert set(a.involved["events"]) == {e["event"] for p in a.qualifying_paths for e in p["events"]}


def test_handoff_followed_by_lineage_qualifies_but_lineage_followed_by_handoff_also_does():
    s = Scn()                                                        # A -(handoff)-> B -(lineage)-> C
    acc = s.accept("A", 100, 500, "203.0.113.7", 51000, 443)
    con = s.connect("A", 150, 500, "B", 40001, 8080, name="nginx")
    rem = s.accept("B", 151, 700, HOSTS["A"][1], 40001, 8080)
    ps = s.pstart("B", 200, 900, 700)
    con2 = s.connect("B", 210, 900, "C", 41000, 22)
    rem2 = s.accept("C", 211, 800, HOSTS["B"][1], 41000, 22)
    s.link(acc, con, "service_handoff", 2, 50)
    s.hop(con, rem)
    s.link(rem, ps, "entry_to_child", 3, 49)
    s.link(ps, con2, "same_process", 2, 10)
    s.hop(con2, rem2)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.SUPPORTED and a.evidence_types == ["process_lineage", "service_handoff"]
    (p,) = a.qualifying_paths
    assert p["n_hops"] == 2 and p["pivot_kinds"] == ["service_handoff", "process_lineage"]
    lm_first, _ = valid_scenario()
    b = lm.analyse_scenario(_lineage_then_handoff())
    assert b.assessment == lm.SUPPORTED and b.qualifying_paths[0]["pivot_kinds"] == ["process_lineage", "service_handoff"]


def _lineage_then_handoff():
    s = Scn()
    chain(s)
    con2 = s.connect("B", 140, 700, "C", 41000, 22)
    rem2 = s.accept("C", 141, 800, HOSTS["B"][1], 41000, 22)
    s.link(3, con2, "service_handoff", 2, 20)
    s.hop(con2, rem2)
    return s.record()


def test_ambiguous_lineage_candidate_blocks_an_informational_service_chain_verdict():
    s = Scn()
    handoff_chain(s, hosts=("B", "C"), t0=300)                       # verified service chain
    chain(s, t_accept=100, t_child=100, t_connect=120)               # lineage candidate with equal timestamps: unresolved
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.INSUFFICIENT and a.reason_code == "ambiguous_event_order"
    assert a.service_chain_paths and any(p["status"] == "ambiguous" for p in a.paths)     # nothing is hidden


def test_ambiguous_service_candidate_does_not_hide_a_verified_service_chain():
    s = Scn()
    handoff_chain(s, hosts=("A", "B"), t0=100)                       # verified
    acc = s.accept("C", 500, 510, "203.0.113.9", 52000, 443)         # equal-timestamp hand-off on another host pair
    con = s.connect("C", 500, 510, "A", 43000, 8080, name="nginx")
    rem = s.accept("A", 500, 520, HOSTS["C"][1], 43000, 8080)
    s.link(acc, con, "service_handoff", 2, 0)
    s.hop(con, rem)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.SERVICE_CHAIN_ONLY and any(p["status"] == "ambiguous" for p in a.paths)


def test_search_bound_with_only_service_chains_is_insufficient_not_informational():
    s = Scn()
    for k in range(3):
        handoff_chain(s, hosts=("A", "B"), t0=100 + 1000 * k, pid=500 + 10 * k, dport0=40000 + 10 * k)
    a = lm.analyse_scenario(s.record(), max_candidates=1)
    assert a.search["truncated"] and a.assessment == lm.INSUFFICIENT and a.reason_code == "search_bound_reached"


# =========================================================================== return to origin
def origin_scenario(entry_source="origin_host"):
    s = Scn()
    if entry_source == "origin_host":                                # B -> A (entry on A), A pivots, A -> B again
        s.connect("B", 90, 600, "A", 33000, 22)
        acc = s.accept("A", 100, 500, HOSTS["B"][1], 33000, 22)
        s.hop(0, acc)
    else:                                                            # same shape but the entry comes from outside
        acc = s.accept("A", 100, 500, "203.0.113.7", 33000, 22)
    ps = s.pstart("A", 110, 900, 500)
    con = s.connect("A", 120, 900, "B", 40000, 22)
    rem = s.accept("B", 120, 700, HOSTS["A"][1], 40000, 22)
    s.link(acc, ps, "entry_to_child", 3, 10)
    s.link(ps, con, "same_process", 2, 10)
    s.hop(con, rem)
    return s


def test_route_that_returns_to_its_entry_source_host_is_not_movement_to_a_new_host():
    a = lm.analyse_scenario(origin_scenario("origin_host").record())
    assert a.assessment == lm.NO_PATH and a.reason_code == "only_route_returns_to_origin"
    assert a.supported_paths == [] and a.search["origin_revisits_rejected"] >= 1
    assert any("origin" in r for r in a.reasoning)


def test_same_route_from_an_outside_source_is_a_progression():
    a = lm.analyse_scenario(origin_scenario("outside").record())
    assert a.assessment == lm.SUPPORTED and a.search["origin_revisits_rejected"] == 0


def test_return_to_an_earlier_host_in_the_middle_of_a_path_is_still_rejected():
    s = Scn()
    chain(s)                                                          # A -> B
    con2 = s.connect("B", 140, 700, "A", 42000, 22, name="sshd")      # B -> back to the entry host A
    rem2 = s.accept("A", 141, 502, HOSTS["B"][1], 42000, 22)
    s.link(3, con2, "service_handoff", 2, 20)
    s.hop(con2, rem2)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.SUPPORTED and all(p["n_hops"] == 1 for p in a.paths)
    assert a.search["host_revisits_rejected"] >= 1


# =========================================================================== flags and stable path IDs
def test_destination_side_activity_is_flagged_but_never_made_positive():
    s = Scn()
    con = s.connect("A", 100, 900, "B", 40000, 22)                    # no predecessor for the connecting event
    rem = s.accept("B", 100, 700, HOSTS["A"][1], 40000, 22)
    ps = s.pstart("B", 110, 901, 700)
    ps2 = s.pstart("B", 120, 902, 901)
    s.hop(con, rem)
    s.link(rem, ps, "entry_to_child", 3, 10)
    s.link(ps, ps2, "parent_child", 2, 10)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.NO_PATH and a.reason_code == "host_to_host_without_pivot_context"
    assert a.flags == ["destination_side_activity_incomplete_predecessor"] and a.supported_paths == []
    assert any("earlier part of a chain may lie outside the scenario" in x for x in a.limitations)
    assert "not proof" in a.assessment_meaning


def test_plain_host_to_host_without_follow_up_has_no_flag():
    s = Scn()
    con = s.connect("A", 100, 900, "B", 40000, 22)
    rem = s.accept("B", 100, 700, HOSTS["A"][1], 40000, 22)
    s.hop(con, rem)
    s.pstart("A", 90, 900, 1)
    s.link(len(s.events) - 1, con, "same_process", 2, 10)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.NO_PATH and a.flags == []


def test_no_supporting_path_wording_is_scoped_to_the_scenario():
    s = Scn()
    acc = s.accept("A", 100, 500, "10.30.1.5", 51000, 22)
    ps = s.pstart("A", 110, 900, 500)
    s.link(acc, ps, "entry_to_child", 3, 10)
    s.pstart("A", 120, 901, 900)
    s.link(1, 2, "parent_child", 2, 10)
    a = lm.analyse_scenario(s.record())
    assert a.assessment == lm.NO_PATH
    assert "within the available scenario evidence" in a.assessment_meaning and "not proof" in a.assessment_meaning
    assert any("not proof" in x for x in a.limitations)


def test_path_ids_are_stable_unique_and_order_independent():
    s = Scn()
    chain(s, t_accept=100, t_child=110, t_connect=120, t_remote=120)
    chain(s, t_accept=200, t_child=210, t_connect=220, t_remote=221)
    rec = s.record()
    a = lm.analyse_scenario(rec)
    ids = [p["path_id"] for p in a.paths]
    assert len(ids) == len(set(ids)) == 2 and all(i.startswith("F2-1-") for i in ids)
    assert [p["path_id"] for p in lm.analyse_scenario(rec).paths] == ids
    shuffled = dataclasses.replace(rec, evidence=rec.evidence.sample(frac=1, random_state=4).reset_index(drop=True),
                                   nodes=rec.nodes.sample(frac=1, random_state=4).reset_index(drop=True))
    assert sorted(p["path_id"] for p in lm.analyse_scenario(shuffled).paths) == sorted(ids)
    other = Scn()
    chain(other, t_accept=100, t_child=110, t_connect=120, t_remote=120)
    assert lm.analyse_scenario(other.record(scenario_id=2)).paths[0]["path_id"].startswith("F2-2-")


# =========================================================================== F1 -> F2 gate hardening
def _mini_handoff(record_f1=None, f1_rows=None, threshold=F1["threshold"], ids=(1,)):
    """An F1F2Handoff whose authoritative f1_scores say what `f1_rows` says (default: the scenario is Suspicious)."""
    s, _ = valid_scenario()
    rec = s.record()
    if record_f1:
        rec.f1.update(record_f1)
    row = {"scenario_id": 1, "anomaly_score": F1["anomaly_score"], "threshold": F1["threshold"], "prediction": "Suspicious",
           "is_suspicious": True, "n_nodes": 4, "n_edges": 3}
    row.update(f1_rows or {})
    scores = pd.DataFrame([row])
    return h.F1F2Handoff(f1_scores=scores, records={1: rec}, normal=scores.iloc[0:0], threshold=threshold)


def test_consistent_handoff_passes_the_gate_and_does_not_change_f1():
    ho = _mini_handoff()
    before = ho.f1_scores.copy()
    assert lm.check_f1_gate(ho) == []
    rep = lm.analyse_handoff(ho)
    assert rep.counts()["analysed"] == 1 and rep.assessments[1].f1 == ho.records[1].f1
    pd.testing.assert_frame_equal(ho.f1_scores, before)                # F2 never touches the authoritative F1 results


def test_forged_record_prediction_and_score_together_are_rejected():
    """The record claims Suspicious with a high score; the authoritative F1 result says Normal with a low score."""
    ho = _mini_handoff(f1_rows={"anomaly_score": 0.4, "prediction": "Normal", "is_suspicious": False})
    assert ho.records[1].f1["prediction"] == "Suspicious" and ho.records[1].f1["anomaly_score"] == 4.0   # forged pair
    with pytest.raises(lm.F1GateError) as e:
        lm.analyse_handoff(ho)
    assert any("did not classify it Suspicious" in p for p in e.value.problems)
    assert any("record prediction" in p for p in e.value.problems) and any("record score" in p for p in e.value.problems)


@pytest.mark.parametrize("record_f1,f1_rows,fragment", [
    ({"anomaly_score": 99.0}, None, "record score"),                          # score inflated, decision kept
    ({"prediction": "Normal"}, None, "record prediction"),                    # decision flipped only
    ({"threshold": 1.0}, None, "record threshold"),                           # threshold altered
    (None, {"threshold": 1.0}, "differs from the handoff's locked threshold"),  # authoritative threshold differs
    (None, {"anomaly_score": 2.0}, "did not classify it Suspicious"),         # authoritative score under the threshold
    (None, {"is_suspicious": False}, "did not classify it Suspicious"),       # authoritative flag says not suspicious
])
def test_each_contradiction_with_the_authoritative_f1_result_is_rejected(record_f1, f1_rows, fragment):
    ho = _mini_handoff(record_f1=record_f1, f1_rows=f1_rows)
    with pytest.raises(lm.F1GateError) as e:
        lm.analyse_handoff(ho)
    assert any(fragment in p for p in e.value.problems), e.value.problems


def test_record_missing_from_f1_results_or_with_wrong_id_is_rejected():
    ho = _mini_handoff(f1_rows={"scenario_id": 2})
    with pytest.raises(lm.F1GateError, match="not present"):
        lm.analyse_handoff(ho)
    ho = _mini_handoff()
    ho.records[1].scenario_id = 7
    with pytest.raises(lm.F1GateError, match="record carries scenario id"):
        lm.analyse_handoff(ho)
    ho = _mini_handoff()
    ho = dataclasses.replace(ho, f1_scores=pd.concat([ho.f1_scores, ho.f1_scores], ignore_index=True))
    with pytest.raises(lm.F1GateError, match="duplicate"):
        lm.analyse_handoff(ho)


def test_gate_runs_before_any_analysis(monkeypatch):
    called = []
    monkeypatch.setattr(lm, "analyse_scenario", lambda *a, **k: called.append(1))
    with pytest.raises(lm.F1GateError):
        lm.analyse_handoff(_mini_handoff(f1_rows={"prediction": "Normal", "is_suspicious": False}))
    assert called == []


def test_real_handoff_passes_the_gate_and_a_forged_real_record_is_rejected(handoff):
    assert lm.check_f1_gate(handoff) == []
    sid = next(iter(handoff.records))
    rec = copy.deepcopy(handoff.records[sid])
    fs = handoff.f1_scores.copy()
    fs.loc[fs.scenario_id == sid, ["anomaly_score", "prediction", "is_suspicious"]] = [0.5, "Normal", False]
    forged = dataclasses.replace(handoff, f1_scores=fs, records={sid: rec})    # record still claims Suspicious + real score
    with pytest.raises(lm.F1GateError):
        lm.analyse_handoff(forged)
    other = next(s for s in handoff.records if s != sid)                       # records with a swapped decision/score pair
    rec2 = copy.deepcopy(handoff.records[other])
    rec2.f1.update(anomaly_score=handoff.records[sid].f1["anomaly_score"] + 1.0)
    with pytest.raises(lm.F1GateError, match="record score"):
        lm.analyse_handoff(dataclasses.replace(handoff, records={other: rec2}))


# =========================================================================== real data: refined distribution
def test_real_scenarios_36595_and_22761_are_classified_as_reviewed(f2):
    a = f2.assessments[36595]
    assert a.assessment == lm.SUPPORTED and a.evidence_types == ["process_lineage", "service_handoff"]
    assert {p["classification"] for p in a.supported_paths} == {"process_lineage_progression", "service_chain_only"}
    b = f2.assessments[22761]
    assert b.assessment == lm.SERVICE_CHAIN_ONLY and b.qualifying_paths == []
    assert max(p["n_hops"] for p in b.supported_paths) == 2 and b.evidence_types == ["service_handoff"]


def test_real_service_chain_only_scenarios_have_no_process_lineage_anywhere(f2):
    svc = [a for a in f2.assessments.values() if a.assessment == lm.SERVICE_CHAIN_ONLY]
    assert len(svc) == 23
    for a in svc:
        assert a.evidence_types == ["service_handoff"] and a.qualifying_paths == []
        assert all(st["link_type"] == "service_handoff" for p in a.supported_paths for s in p["segments"] for st in s["pivot_links"])
        assert all(s["pivot_kind"] == "service_handoff" for p in a.supported_paths for s in p["segments"])


def test_real_potential_movement_scenarios_all_contain_a_process_lineage_pivot(f2):
    sup = [a for a in f2.assessments.values() if a.assessment == lm.SUPPORTED]
    assert len(sup) == 141
    for a in sup:
        assert a.qualifying_paths and "process_lineage" in a.evidence_types
        assert any(st["link_type"] in ("entry_to_child", "parent_child") for p in a.qualifying_paths
                   for s in p["segments"] for st in s["pivot_links"])


def test_real_destination_side_activity_cases_are_flagged_and_stay_negative(f2):
    flagged = [a for a in f2.assessments.values() if a.flags]
    assert len(flagged) == 95
    for a in flagged:
        assert a.assessment == lm.NO_PATH and a.reason_code == "host_to_host_without_pivot_context" and a.supported_paths == []
        assert a.flags == ["destination_side_activity_incomplete_predecessor"]


def test_real_path_ids_are_unique_and_f1_is_untouched(f2, handoff):
    ids = [p["path_id"] for a in f2.assessments.values() for p in a.paths]
    assert len(ids) == len(set(ids)) > 0
    auth = handoff.f1_scores.set_index("scenario_id")
    for sid, a in f2.assessments.items():
        assert a.f1["anomaly_score"] == float(auth.at[sid, "anomaly_score"]) and a.f1["prediction"] == auth.at[sid, "prediction"]
    assert not any(a.search["origin_revisits_rejected"] for a in f2.assessments.values())
