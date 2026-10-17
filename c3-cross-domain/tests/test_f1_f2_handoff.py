"""Tests for the F1 -> F2 handoff (xdr_c3.f1_f2_handoff) on the real, previously generated scenario results."""
import ast
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xdr_c3 import f1_f2_handoff as h
from xdr_c3 import stage2_validate as s2
from xdr_c3.config import PROJECT_ROOT

GROUND_TRUTH = {"label", "type", "attack_type"}
SAVED_SCORES = PROJECT_ROOT / "verification" / "stage5" / "scenario_scores_all_graphs.csv"


# =========================================================================== F1 -> F2 gating
def test_only_suspicious_scenarios_get_records(handoff, result):
    s = result.scores
    assert set(handoff.records) == set(s.loc[s.is_suspicious, "scenario_id"]) and len(handoff.records) == 1499
    assert len(handoff.normal) == 11281 and handoff.summary() == {
        "scenarios_scored": 12780, "suspicious_handed_to_f2": 1499, "normal_retained_without_f2": 11281,
        "threshold": 2.5825197985552437}
    assert not set(handoff.normal["scenario_id"]) & set(handoff.records)
    assert (handoff.normal["prediction"] == "Normal").all()
    assert (handoff.normal["f2_status"] == "not_analysed (F1 Normal)").all()


def test_normal_scenarios_get_no_detailed_processing(s2_data, s3b, small_result, detector, monkeypatch):
    calls = []
    real = h._observed_facts
    monkeypatch.setattr(h, "_observed_facts", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    ho = h.build_f1_f2_handoff(s2_data, s3b, small_result, detector.contract)
    n_susp = int(small_result.scores["is_suspicious"].sum())
    assert len(calls) == len(ho.records) == n_susp == 12            # detailed work done only for Suspicious scenarios
    assert len(ho.normal) == 12 and len(ho.f1_scores) == 24
    normal_ids = set(small_result.scores.loc[~small_result.scores.is_suspicious, "scenario_id"])
    assert not normal_ids & set(ho.records)


def test_selection_follows_actual_model_decisions_without_recalculation(handoff, result):
    s = result.scores
    assert np.array_equal(h.select_for_f2(result), s.loc[s["anomaly_score"] > 2.5825197985552437, "scenario_id"].to_numpy())
    for sid, rec in list(handoff.records.items())[::50]:
        row = s[s.scenario_id == sid].iloc[0]
        assert rec.f1["anomaly_score"] == row["anomaly_score"] and rec.f1["threshold"] == 2.5825197985552437
        assert rec.f1["prediction"] == "Suspicious" and rec.f1["anomaly_score"] > rec.f1["threshold"]
        assert (rec.f1["method"], rec.f1["k"], rec.f1["checkpoint_epoch"]) == ("knn_embedding", 5, 40)
    assert handoff.f1_scores.equals(s)                               # F1 result passed through unchanged


def test_selection_matches_previously_saved_stage5_scores(handoff):
    saved = pd.read_csv(SAVED_SCORES)
    assert set(handoff.records) == set(saved.loc[saved.prediction == "Suspicious", "scenario_id"])
    assert (saved.loc[saved.is_suspicious, "anomaly_score"] > 2.5825197985552437).all()


def test_inconsistent_prediction_column_is_rejected(result):
    bad = dataclasses.replace(result, scores=result.scores.assign(prediction="Normal"))
    with pytest.raises(h.HandoffIntegrityError):
        h.select_for_f2(bad)


# =========================================================================== identities, timestamps, attributes
def test_event_ids_and_timestamps_preserved(handoff, s2_data, graphs):
    host, iot = s2_data.to_stage3_frames()
    for sid, rec in handoff.records.items():
        nd = rec.nodes
        ref_nodes = graphs.nodes[graphs.nodes.scenario_id == sid].reset_index(drop=True)
        assert list(nd["node"]) == list(ref_nodes["node"]) and list(nd["row"]) == list(ref_nodes["row"])
        assert list(nd["local_index"]) == list(range(len(nd)))                         # stable graph-local indices
        assert (nd["node"] == nd["kind"] + ":" + nd["row"].astype(str)).all()
        hr, ir = nd.loc[nd.kind == "host", "row"].to_numpy(), nd.loc[nd.kind == "iot", "row"].to_numpy()
        assert np.array_equal(nd.loc[nd.kind == "host", "timestamp"].to_numpy(object), host["timestamp"].to_numpy(object)[hr])
        assert np.array_equal(nd.loc[nd.kind == "iot", "timestamp"].to_numpy(object), iot["timestamp"].to_numpy(object)[ir])
        assert np.array_equal(nd["ts_s"].to_numpy(), ref_nodes["ts_s"].to_numpy())


def test_original_event_attributes_are_unchanged_and_label_free(handoff, s2_data):
    host, iot = s2_data.to_stage3_frames()
    for sid, rec in list(handoff.records.items())[::40]:
        he, ie = rec.host_events, rec.iot_events
        assert not GROUND_TRUTH & (set(he.columns) | set(ie.columns) | set(rec.nodes.columns))
        pd.testing.assert_frame_equal(he.drop(columns="local_index").reset_index(drop=True),
                                      host.iloc[he["row"].to_numpy()].drop(columns="ts").reset_index(drop=True))
        pd.testing.assert_frame_equal(ie.drop(columns="local_index").reset_index(drop=True),
                                      iot.iloc[ie["row"].to_numpy()].drop(columns="ts").reset_index(drop=True))
        for col in ("host_id", "host_ip", "process_id", "process_name", "event_type", "src_ip", "dst_ip", "protocol"):
            assert col in he.columns
        for col in ("iot_device_id", "src_ip", "dst_ip", "service", "protocol"):
            assert col in ie.columns
    assert set(rec.component_columns["host"]) <= set(rec.host_events.columns)      # component outputs are labelled as such


# =========================================================================== graph consistency
def test_nodes_and_edges_consistent_with_scenario_graph(handoff, result, graphs):
    g = result.graphs
    pos = {int(s): i for i, s in enumerate(g.scenario_id)}
    arrays = graphs.arrays()
    gpos = {int(s): i for i, s in enumerate(arrays["scenario_id"])}
    for sid, rec in handoff.records.items():
        i = pos[sid]
        n0, n1, e0, e1 = g.node_ptr[i], g.node_ptr[i + 1], g.edge_ptr[i], g.edge_ptr[i + 1]
        half = (e1 - e0) // 2
        assert len(rec.nodes) == n1 - n0 and len(rec.edges) == half
        ei = g.edge_index[:, e0:e1]
        assert np.array_equal(rec.edges["local_a"], ei[0, :half]) and np.array_equal(rec.edges["local_b"], ei[1, :half])
        assert np.array_equal(ei[:, half:], ei[::-1, :half])                           # both directions stored
        j = gpos[sid]                                                                  # and equals the Stage 4 arrays
        assert np.array_equal(arrays["edge_index"][:, arrays["edge_ptr"][j]:arrays["edge_ptr"][j] + half], ei[:, :half])
        names = rec.nodes["node"].to_numpy()
        assert np.array_equal(rec.edges["node_a"], names[rec.edges["local_a"]])
        assert np.array_equal(rec.edges["node_b"], names[rec.edges["local_b"]])
        assert rec.f2_ready and rec.integrity_problems == []


def test_edge_relation_matches_model_input_encoding(handoff, result, detector):
    g = result.graphs
    rel_names = [r[len("e_type="):] for r in detector.contract.relation_names]
    for sid, rec in list(handoff.records.items())[::25]:
        i = int(np.flatnonzero(g.scenario_id == sid)[0])
        e0 = g.edge_ptr[i]
        half = (g.edge_ptr[i + 1] - e0) // 2
        decoded = [rel_names[k] for k in g.edge_attr[e0:e0 + half, 4:14].argmax(axis=1)]
        assert list(rec.edges["relation"]) == decoded


# =========================================================================== evidence tracing / no fabrication
def test_every_edge_traces_to_original_stage3b_links(handoff, s3b):
    L = s3b.links
    universe = set(zip(L.node_a, L.node_b, L.link_type, L.points, L.gap_sec))
    total_edges = 0
    for sid, rec in handoff.records.items():
        ev, ed = rec.evidence, rec.edges
        assert set(zip(ev.node_a, ev.node_b, ev.link_type, ev.points, ev.gap_sec)) <= universe
        assert set(ed["edge_index"]) == set(ev["edge_index"].astype(int))              # every edge has evidence
        pair = set(zip(ed.local_a, ed.local_b))
        assert set(zip(ev.local_a, ev.local_b)) == pair                                # no evidence without an edge
        merged = ev.merge(ed[["edge_index", "relation"]], on="edge_index")
        assert (merged["link_type"] == merged["relation"]).all()
        total_edges += len(ed)
    assert total_edges > 0


def test_evidence_is_complete_for_each_scenario(handoff, s3b):
    sc = dict(zip(s3b.assignments["node"], s3b.assignments["scenario_id"]))
    L = s3b.links
    same = L["node_a"].map(sc) == L["node_b"].map(sc)
    by_scn = L[same].groupby(L[same]["node_a"].map(sc)).size()
    for sid, rec in list(handoff.records.items())[::30]:
        assert len(rec.evidence) == by_scn[sid]                                        # no link dropped, none added


def test_no_fabricated_relationships_or_findings(handoff, graphs):
    arrays = graphs.arrays()
    n_logical = {int(s): (arrays["edge_ptr"][i + 1] - arrays["edge_ptr"][i]) // 2 for i, s in enumerate(arrays["scenario_id"])}
    for sid, rec in handoff.records.items():
        assert len(rec.edges) == n_logical[sid] and not rec.edges.duplicated(["local_a", "local_b"]).any()
        assert rec.assessment["lateral_movement"] == "not_assessed"
        assert set(rec.observed) >= {"n_events", "relation_counts", "iot_host_link_present", "host_to_host_link_present"}
        assert rec.observed["iot_host_link_present"] == bool((rec.edges["relation"] == "iot_host_match").any())
        assert rec.observed["n_edges"] == len(rec.edges) and rec.observed["n_events"] == len(rec.nodes)
    flat = json.dumps(handoff.records[next(iter(handoff.records))].to_dict())
    for forbidden in ("is_lateral", "lateral_movement_detected", "attack_type", '"label"'):
        assert forbidden not in flat


def test_model_signals_are_kept_distinct_from_evidence(handoff):
    rec = next(iter(handoff.records.values()))
    assert {"model_node_error"} <= set(rec.nodes.columns)
    assert {"model_edge_error_forward", "model_edge_error_reverse", "model_edge_error_mean"} <= set(rec.edges.columns)
    assert not any(c.startswith("model_") for c in rec.evidence.columns)               # evidence carries no model output
    assert "anomaly_score" in rec.f1 and "graph_errors" in rec.model_signals
    assert "not confirmed security evidence" in rec.model_signals["note"]
    assert "not_assessed" in json.dumps(rec.assessment) and "not proof" in rec.assessment["note"]
    assert all(v for v in h.FIELD_PROVENANCE.values())
    assert "model_signal" in h.FIELD_PROVENANCE["f1.anomaly_score / threshold / prediction"]
    pair = (rec.edges["model_edge_error_forward"] + rec.edges["model_edge_error_reverse"]) / 2
    assert np.allclose(pair, rec.edges["model_edge_error_mean"])


def test_record_errors_come_from_the_model_output(handoff, result):
    g = result.graphs
    for sid, rec in list(handoff.records.items())[::60]:
        i = int(np.flatnonzero(g.scenario_id == sid)[0])
        n0, n1, e0 = g.node_ptr[i], g.node_ptr[i + 1], g.edge_ptr[i]
        assert np.array_equal(rec.nodes["model_node_error"].to_numpy(), result.node_error[n0:n1])
        half = (g.edge_ptr[i + 1] - e0) // 2
        assert np.array_equal(rec.edges["model_edge_error_forward"].to_numpy(), result.edge_error[e0:e0 + half])
        assert np.array_equal(rec.edges["model_edge_error_reverse"].to_numpy(), result.edge_error[e0 + half:e0 + 2 * half])
        ge = result.graph_errors.iloc[i]
        assert rec.model_signals["graph_errors"]["recon_total"] == float(ge["recon_total"])


# =========================================================================== ground-truth independence
def test_handoff_cannot_depend_on_ground_truth(s2_data, s3b, small_result, detector):
    base = h.build_f1_f2_handoff(s2_data, s3b, small_result, detector.contract)
    rng = np.random.default_rng(11)
    shuffled = s2.Stage2Data(s2_data.host, s2_data.iot,
                             s2_data.host_truth.iloc[rng.permutation(len(s2_data.host_truth))].set_axis(s2_data.host_truth.index),
                             s2_data.iot_truth.iloc[rng.permutation(len(s2_data.iot_truth))].set_axis(s2_data.iot_truth.index),
                             s2_data.report)
    other = h.build_f1_f2_handoff(shuffled, s3b, small_result, detector.contract)
    assert base.suspicious_ids == other.suspicious_ids
    for sid in base.suspicious_ids:
        assert base.records[sid].to_dict() == other.records[sid].to_dict()
    # removed ground truth: frames without any truth columns give the same event tables
    h_ = s2_data.host.drop(columns=["ts", "row"]).astype({"process_id": str, "parent_process_id": str})
    unlabelled = s2.validate_and_build(h_, s2_data.iot.drop(columns=["ts", "row"]), allow_missing_ground_truth=True)
    third = h.build_f1_f2_handoff(unlabelled, s3b, small_result, detector.contract)
    for sid in base.suspicious_ids:
        assert base.records[sid].to_dict() == third.records[sid].to_dict()


def test_handoff_code_has_no_ground_truth_inputs():
    tree = ast.parse(Path(h.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in GROUND_TRUTH | {"node_label", "is_attack"}:
            raise AssertionError(node.value)
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        assert not {a.arg for a in fn.args.args} & {"labels", "truth", "node_label", "scenario_is_attack"}, fn.name


# =========================================================================== missing / inconsistent evidence is detected
def _build(s2_data, s3b, small_result, detector, strict=True):
    return h.build_f1_f2_handoff(s2_data, s3b, small_result, detector.contract, strict=strict)


def _a_suspicious_edge(s3b, small_result):
    s = small_result.scores
    sid = int(s.loc[s.is_suspicious, "scenario_id"].iloc[0])
    sc = dict(zip(s3b.assignments["node"], s3b.assignments["scenario_id"]))
    idx = s3b.links.index[(s3b.links.node_a.map(sc) == sid) & (s3b.links.node_b.map(sc) == sid)]
    return sid, idx


def test_missing_relationship_evidence_is_detected(s2_data, s3b, small_result, detector):
    sid, idx = _a_suspicious_edge(s3b, small_result)
    stripped = dataclasses.replace(s3b, links=s3b.links.drop(index=idx[:1]))
    with pytest.raises(h.HandoffIntegrityError, match="no relationship evidence"):
        _build(s2_data, stripped, small_result, detector)
    lenient = _build(s2_data, stripped, small_result, detector, strict=False)
    assert not lenient.records[sid].f2_ready and any("no relationship evidence" in p for p in lenient.records[sid].integrity_problems)


def test_contradicting_evidence_type_is_detected(s2_data, s3b, small_result, detector):
    sid, idx = _a_suspicious_edge(s3b, small_result)
    links = s3b.links.copy()
    other = "same_process" if links.loc[idx[0], "link_type"] != "same_process" else "parent_child"
    links.loc[idx[0], "link_type"] = other
    with pytest.raises(h.HandoffIntegrityError, match="evidence"):
        _build(s2_data, dataclasses.replace(s3b, links=links), small_result, detector)


def test_tampered_timestamp_is_detected(s2_data, s3b, small_result, detector):
    sid = int(small_result.scores.loc[small_result.scores.is_suspicious, "scenario_id"].iloc[0])
    row = int(small_result.nodes.loc[(small_result.nodes.scenario_id == sid) & (small_result.nodes.kind == "host"), "row"].iloc[0])
    host = s2_data.host.copy()
    host.loc[host["row"] == row, "timestamp"] = "01/01/2000 00:00:00"
    bad = dataclasses.replace(s2_data, host=host)
    with pytest.raises(h.HandoffIntegrityError, match="timestamps"):
        _build(bad, s3b, small_result, detector)


def test_edge_attribute_inconsistent_with_events_is_detected(s2_data, s3b, small_result, detector):
    g = small_result.graphs
    i = int(np.flatnonzero(small_result.scores["is_suspicious"].to_numpy())[0])      # a graph that enters F2
    e0 = int(g.edge_ptr[i])
    ea = g.edge_attr.copy()
    ea[e0, 2] = 1.0 - ea[e0, 2]                                      # flip the stored same_host flag of its first edge
    bad = dataclasses.replace(small_result, graphs=dataclasses.replace(g, edge_attr=ea))
    with pytest.raises(h.HandoffIntegrityError, match="same_host"):
        _build(s2_data, s3b, bad, detector)


def test_misaligned_node_references_are_rejected(s2_data, s3b, small_result, detector):
    shuffled = small_result.nodes.sample(frac=1, random_state=3).reset_index(drop=True)
    with pytest.raises(h.HandoffIntegrityError, match="aligned"):
        _build(s2_data, s3b, dataclasses.replace(small_result, nodes=shuffled), detector)
    with pytest.raises(ValueError):
        _build(s2_data, s3b, dataclasses.replace(small_result, nodes=None), detector)


# =========================================================================== serialisation
def test_records_are_json_serialisable(handoff):
    rec = handoff.records[next(iter(handoff.records))]
    d = json.loads(json.dumps(rec.to_dict()))
    assert d["scenario_id"] == rec.scenario_id and d["f1"]["prediction"] == "Suspicious"
    assert len(d["nodes"]) == len(rec.nodes) and len(d["edges"]) == len(rec.edges) and len(d["evidence"]) == len(rec.evidence)
    assert d["assessment"]["lateral_movement"] == "not_assessed" and "field_provenance" in d
