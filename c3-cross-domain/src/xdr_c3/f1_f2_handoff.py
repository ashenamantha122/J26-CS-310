"""F1 -> F2 internal handoff.

Function 1 (the saved R-GCN-GAE, `inference.Stage5Detector`) decides per scenario graph: Normal or Suspicious.
Only scenarios the model classified Suspicious are handed to the detailed Function 2 analysis; Normal scenarios keep
their F1 result and get no detailed record. Nothing here recomputes a score, changes the locked threshold or looks at
ground truth: the selection is `InferenceResult.scores["is_suspicious"]` and nothing else.

A handoff record keeps four kinds of information strictly apart (see `FIELD_PROVENANCE`):
  * identifiers      - scenario id, graph-local node index, `host:<row>` / `iot:<row>` event ids
  * observed telemetry - the original Host / IoT event attributes exactly as read in Stage 2 (no label columns)
  * rule-derived relationships - the Stage 3 / 3b links (type, points, gap) that produced each graph edge
  * model signals    - anomaly score, reconstruction errors. These are NOT security evidence: a high error or a
                       connected graph path is not proof of lateral movement. `assessment` stays `not_assessed`.
Nothing is invented: attributes come from the Stage 2 frames, edges from the Stage 4 graph arrays, relationship
evidence from the Stage 3b link table, and any inconsistency is reported (`HandoffIntegrityError`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .inference import SUSPICIOUS, InferenceResult
from .model_contract import ModelContract, load_contract
from .stage2_validate import FMT

SHORT_GAP_S = 300                      # Stage 4 definition of the short-gap flag (reported, re-verified below)

# Component outputs are predictions of Components 1 / 2, not raw telemetry; they are kept but labelled.
HOST_COMPONENT_COLUMNS = ("trace_mode", "threat_confidence_score", "threat_state")
IOT_COMPONENT_COLUMNS = ("confidence_score", "predicted_class")

FIELD_PROVENANCE = {
    "scenario_id": "identifier", "local_index": "identifier", "node": "identifier", "row": "identifier",
    "timestamp": "observed_telemetry", "ts_s": "observed_telemetry",
    "host_events / iot_events (all columns)": "observed_telemetry",
    "host_events component columns": "component_output (Component 1 prediction, not raw telemetry)",
    "iot_events component columns": "component_output (Component 2 prediction, not raw telemetry)",
    "edges.local_a / local_b / node_a / node_b": "identifier",
    "edges.relation": "rule_derived_relationship (Stage 4 one-hot of the Stage 3/3b link type)",
    "evidence.link_type / points / gap_sec": "rule_derived_relationship (Stage 3 / 3b rule output)",
    "nodes.host_ip": "identifier (for an IoT flow: the managed host at one end of the flow, as derived in Stage 3b / 4)",
    "edges.same_host / short_gap / e_score / min_gap_sec": "rule_derived_relationship (Stage 4 edge attributes, unscaled with the saved scaler)",
    "f1.anomaly_score / threshold / prediction": "model_signal",
    "nodes.model_node_error / edges.model_edge_error_*": "model_signal (reconstruction error; not security evidence)",
    "model_signals.graph_errors": "model_signal",
    "assessment": "none (Function 2 has not run; no lateral-movement finding is made here)",
}


class HandoffIntegrityError(ValueError):
    """The scenario graph, its events or its relationship evidence are missing or inconsistent."""

    def __init__(self, problems):
        self.problems = problems
        super().__init__(f"{len(problems)} handoff integrity problem(s): " + "; ".join(problems[:6]))


@dataclass
class ScenarioHandoff:
    scenario_id: int
    f1: dict                       # anomaly_score, threshold, prediction, method, k, checkpoint epoch
    model_signals: dict            # graph-level reconstruction errors (model output, not evidence)
    nodes: pd.DataFrame            # local_index, node, kind, row, timestamp, ts_s, host_ip, model_node_error
    host_events: pd.DataFrame      # local_index + every original Host column (no ground truth)
    iot_events: pd.DataFrame       # local_index + every original IoT column (no ground truth)
    edges: pd.DataFrame            # one row per logical edge, graph-local endpoints + observed flags + model errors
    evidence: pd.DataFrame         # one row per Stage 3/3b link behind an edge
    observed: dict                 # directly countable facts (hosts, devices, cross-domain link present, ...)
    component_columns: dict
    integrity_problems: list = field(default_factory=list)
    assessment: dict = field(default_factory=lambda: {
        "lateral_movement": "not_assessed",
        "note": "Function 2 has not been applied; scores and reconstruction errors are not proof of lateral movement."})

    @property
    def f2_ready(self):
        return not self.integrity_problems

    def to_dict(self):
        def recs(df):
            return json.loads(df.to_json(orient="records"))
        return {"scenario_id": int(self.scenario_id), "f1": self.f1, "model_signals": self.model_signals,
                "nodes": recs(self.nodes), "host_events": recs(self.host_events), "iot_events": recs(self.iot_events),
                "edges": recs(self.edges), "evidence": recs(self.evidence), "observed": self.observed,
                "component_columns": self.component_columns, "assessment": self.assessment,
                "integrity_problems": self.integrity_problems, "field_provenance": FIELD_PROVENANCE}


@dataclass
class F1F2Handoff:
    """Result of the F1 -> F2 gate for one inference run."""
    f1_scores: pd.DataFrame        # every scenario: F1 result (unchanged model output)
    records: dict                  # scenario_id -> ScenarioHandoff, ONLY for Suspicious scenarios
    normal: pd.DataFrame           # Normal scenarios: F1 result retained, no detailed F2 processing
    threshold: float

    @property
    def suspicious_ids(self):
        return list(self.records)

    def summary(self):
        return {"scenarios_scored": len(self.f1_scores), "suspicious_handed_to_f2": len(self.records),
                "normal_retained_without_f2": len(self.normal), "threshold": self.threshold}


# --------------------------------------------------------------------------- selection
def select_for_f2(result: InferenceResult):
    """Scenario ids the saved model classified Suspicious (the only ones that enter detailed Function 2 analysis)."""
    s = result.scores
    flagged = s["is_suspicious"].to_numpy()
    if not np.array_equal(flagged, (s["prediction"] == SUSPICIOUS).to_numpy()):
        raise HandoffIntegrityError(["prediction column disagrees with is_suspicious"])
    return s.loc[flagged, "scenario_id"].to_numpy(dtype=np.int64)


# --------------------------------------------------------------------------- building
def build_f1_f2_handoff(data, stage3b, result: InferenceResult, contract: ModelContract = None,
                        strict: bool = True) -> F1F2Handoff:
    """Create handoff records for the Suspicious scenarios of `result` (an `InferenceResult` from
    `Stage5Detector.predict_collection`). `data` is the Stage 2 `Stage2Data` (its frames hold no ground truth),
    `stage3b` the `Stage3bResult` whose combined links are the relationship evidence.
    With `strict=True` any integrity problem raises `HandoffIntegrityError`; otherwise the record is created with
    `integrity_problems` filled in and `f2_ready == False`."""
    if result.nodes is None:
        raise ValueError("the inference result carries no node references; use Stage5Detector.predict_collection")
    contract = contract or load_contract()
    scores = result.scores
    g = result.graphs
    nodes_all = result.nodes
    # the node reference table must line up with the graph arrays
    expected_sid = np.repeat(g.scenario_id, g.n_nodes)
    if len(nodes_all) != len(g.x) or not np.array_equal(nodes_all["scenario_id"].to_numpy(), expected_sid):
        raise HandoffIntegrityError(["node references are not aligned with the graph arrays"])

    host, iot = data.to_stage3_frames()                   # label-free frames; row == original 0-based file position
    flagged = select_for_f2(result)
    pos_of = {int(s): i for i, s in enumerate(g.scenario_id)}

    links = _evidence_links(stage3b.links, nodes_all, set(flagged.tolist()), g, pos_of)
    rel_names = list(contract.relation_names)
    rel_cols = list(contract.relation_cols)
    e_mu, e_sd = contract.edge_scaler["e_score"]
    g_mu, g_sd = contract.edge_scaler["e_log_gap"]
    host_cols = [c for c in host.columns if c not in ("ts",)]
    iot_cols = [c for c in iot.columns if c not in ("ts",)]

    records = {}
    for sid in flagged:
        i = pos_of[int(sid)]
        n0, n1 = int(g.node_ptr[i]), int(g.node_ptr[i + 1])
        e0, e1 = int(g.edge_ptr[i]), int(g.edge_ptr[i + 1])
        half = (e1 - e0) // 2
        nd = nodes_all.iloc[n0:n1]
        n = n1 - n0
        local = np.arange(n)

        host_rows = nd.loc[nd["kind"] == "host", "row"].to_numpy()
        iot_rows = nd.loc[nd["kind"] == "iot", "row"].to_numpy()
        h_local, i_local = local[(nd["kind"] == "host").to_numpy()], local[(nd["kind"] == "iot").to_numpy()]
        host_ev = host.iloc[host_rows][host_cols].reset_index(drop=True)
        host_ev.insert(0, "local_index", h_local)
        iot_ev = iot.iloc[iot_rows][iot_cols].reset_index(drop=True)
        iot_ev.insert(0, "local_index", i_local)
        ts_str = np.empty(n, dtype=object)
        ts_str[h_local] = host_ev["timestamp"].to_numpy(object)
        ts_str[i_local] = iot_ev["timestamp"].to_numpy(object)
        nodes = pd.DataFrame({"local_index": local, "node": nd["node"].to_numpy(), "kind": nd["kind"].to_numpy(),
                              "row": nd["row"].to_numpy(), "timestamp": ts_str, "ts_s": nd["ts_s"].to_numpy(),
                              "host_ip": nd["host_ip"].to_numpy(), "model_node_error": result.node_error[n0:n1]})

        ei = g.edge_index[:, e0:e1]
        fwd, rev = ei[:, :half], ei[:, half:]
        attr = g.edge_attr[e0:e0 + half]
        err = result.edge_error[e0:e1]
        rel_idx = attr[:, rel_cols].argmax(axis=1)
        edges = pd.DataFrame({
            "edge_index": np.arange(half), "local_a": fwd[0], "local_b": fwd[1],
            "node_a": nodes["node"].to_numpy()[fwd[0]], "node_b": nodes["node"].to_numpy()[fwd[1]],
            "relation": [rel_names[k][len("e_type="):] for k in rel_idx],
            "e_score": np.round(attr[:, contract.edge_feature_names.index("e_score")].astype(np.float64) * e_sd + e_mu, 3),
            "min_gap_sec": np.round(np.expm1(attr[:, contract.edge_feature_names.index("e_log_gap")].astype(np.float64) * g_sd + g_mu), 3),
            "same_host": attr[:, contract.edge_feature_names.index("e_same_host")].astype(int),
            "short_gap": attr[:, contract.edge_feature_names.index("e_short_gap")].astype(int),
            "model_edge_error_forward": err[:half], "model_edge_error_reverse": err[half:]})
        edges["model_edge_error_mean"] = 0.5 * (edges["model_edge_error_forward"] + edges["model_edge_error_reverse"])

        ev = links.get(int(sid))
        if ev is None:
            ev = pd.DataFrame(columns=["local_a", "local_b", "node_a", "node_b", "link_type", "points", "gap_sec"])
        ev = ev.reset_index(drop=True)
        ev.insert(0, "scenario_id", int(sid))
        ev = ev.merge(edges[["edge_index", "local_a", "local_b"]], on=["local_a", "local_b"], how="left")
        ev = ev[["scenario_id", "edge_index", "local_a", "local_b", "node_a", "node_b", "link_type", "points", "gap_sec"]]

        sc = scores.iloc[i]
        row_err = result.graph_errors.iloc[i]
        rec = ScenarioHandoff(
            scenario_id=int(sid),
            f1={"anomaly_score": float(sc["anomaly_score"]), "threshold": float(sc["threshold"]),
                "prediction": str(sc["prediction"]), "method": "knn_embedding", "k": 5,
                "model": "R-GCN-GAE", "checkpoint_epoch": 40},
            model_signals={"graph_errors": {k: float(row_err[k]) for k in row_err.index},
                           "note": "reconstruction errors are model output, not confirmed security evidence"},
            nodes=nodes, host_events=host_ev, iot_events=iot_ev, edges=edges, evidence=ev,
            observed=_observed_facts(nodes, host_ev, iot_ev, edges),
            component_columns={"host": list(HOST_COMPONENT_COLUMNS), "iot": list(IOT_COMPONENT_COLUMNS)})
        rec.integrity_problems = validate_record(rec, int(sc["n_nodes"]), int(sc["n_edges"]), contract)
        if rec.integrity_problems and strict:
            raise HandoffIntegrityError([f"scenario {sid}: {p}" for p in rec.integrity_problems])
        records[int(sid)] = rec

    normal = scores.loc[~scores["is_suspicious"], ["scenario_id", "anomaly_score", "threshold", "prediction"]].copy()
    normal["f2_status"] = "not_analysed (F1 Normal)"
    return F1F2Handoff(scores.copy(), records, normal.reset_index(drop=True), float(scores["threshold"].iloc[0]))


def _evidence_links(links, nodes_all, wanted, g, pos_of):
    """Stage 3b links whose two events lie in the same Suspicious scenario, keyed by scenario id, with
    graph-local endpoints. Links are taken as they are (original direction, points, gap)."""
    local = nodes_all.groupby("scenario_id").cumcount().to_numpy()
    keyed = pd.DataFrame({"scenario_id": nodes_all["scenario_id"].to_numpy(), "local": local},
                         index=nodes_all["node"].to_numpy())
    keyed = keyed[keyed["scenario_id"].isin(wanted)]
    a = keyed.reindex(links["node_a"])
    b = keyed.reindex(links["node_b"])
    ok = a["scenario_id"].notna().to_numpy() & (a["scenario_id"].to_numpy() == b["scenario_id"].to_numpy())
    sel = links[ok].copy()
    sel["scenario_id"] = a["scenario_id"].to_numpy()[ok].astype(np.int64)
    la, lb = a["local"].to_numpy()[ok].astype(np.int64), b["local"].to_numpy()[ok].astype(np.int64)
    sel["local_a"], sel["local_b"] = np.minimum(la, lb), np.maximum(la, lb)       # Stage 4 edge orientation (lo < hi)
    sel = sel[["scenario_id", "local_a", "local_b", "node_a", "node_b", "link_type", "points", "gap_sec"]]
    return {int(s): d.drop(columns="scenario_id") for s, d in sel.groupby("scenario_id", sort=False)}


def _observed_facts(nodes, host_ev, iot_ev, edges):
    """Plain counts / sets read directly from the records. They are facts about the data, not findings."""
    rel = edges["relation"].value_counts().to_dict()
    return {"n_events": int(len(nodes)), "n_host_events": int(len(host_ev)), "n_iot_flows": int(len(iot_ev)),
            "n_edges": int(len(edges)), "relation_counts": {k: int(v) for k, v in rel.items()},
            "host_ips_of_events": sorted({h for h in nodes["host_ip"] if h}),
            "host_ids": sorted(set(host_ev["host_id"])) if len(host_ev) else [],
            "iot_devices": sorted(set(iot_ev["iot_device_id"])) if len(iot_ev) else [],
            "iot_host_link_present": bool(rel.get("iot_host_match", 0) > 0),
            "host_to_host_link_present": bool(rel.get("connect_accept", 0) > 0),
            "first_timestamp": str(nodes["timestamp"].iloc[int(nodes["ts_s"].idxmin())]),
            "last_timestamp": str(nodes["timestamp"].iloc[int(nodes["ts_s"].idxmax())])}


# --------------------------------------------------------------------------- integrity
def validate_record(rec: ScenarioHandoff, n_nodes_expected, n_edges_expected, contract: ModelContract):
    """Return a list of problems (empty = consistent). Checks identity, events, edges and evidence tracing."""
    p = []
    nd, ed, ev = rec.nodes, rec.edges, rec.evidence
    n = len(nd)
    if n != n_nodes_expected:
        p.append(f"node count {n} != F1 result {n_nodes_expected}")
    if len(ed) != n_edges_expected:
        p.append(f"edge count {len(ed)} != F1 result {n_edges_expected}")
    if not np.array_equal(nd["local_index"].to_numpy(), np.arange(n)):
        p.append("graph-local node indices are not 0..n-1")
    if not nd["node"].is_unique or not (nd["node"] == nd["kind"] + ":" + nd["row"].astype(str)).all():
        p.append("event ids are not unique or do not equal kind:row")
    if len(rec.host_events) + len(rec.iot_events) != n:
        p.append("events missing for some nodes")
    else:
        for kind, frame in (("host", rec.host_events), ("iot", rec.iot_events)):
            sub = nd[nd["kind"] == kind]
            if not np.array_equal(sub["local_index"].to_numpy(), frame["local_index"].to_numpy()):
                p.append(f"{kind} events do not line up with graph nodes")
            elif not np.array_equal(sub["timestamp"].to_numpy(object), frame["timestamp"].to_numpy(object)):
                p.append(f"{kind} timestamps differ between nodes and events")
    parsed = pd.to_datetime(nd["timestamp"], format=FMT, errors="coerce")
    if parsed.isna().any() or not np.array_equal(parsed.astype("datetime64[s]").astype("int64").to_numpy(), nd["ts_s"].to_numpy()):
        p.append("original timestamps do not match the graph node times")
    if len(ed):
        if ((ed["local_a"] >= ed["local_b"]) | (ed["local_a"] < 0) | (ed["local_b"] >= n)).any():
            p.append("edge endpoints are out of range or not ordered")
        if ed.duplicated(["local_a", "local_b"]).any():
            p.append("duplicate edges")
        lookup = nd["node"].to_numpy()
        if not (np.array_equal(ed["node_a"].to_numpy(), lookup[ed["local_a"]]) and
                np.array_equal(ed["node_b"].to_numpy(), lookup[ed["local_b"]])):
            p.append("edge node ids do not match graph-local indices")
    # every edge must trace back to at least one Stage 3 / 3b link with a type equal to the edge relation
    if len(ed):
        unmatched = set(ed["edge_index"]) - set(ev["edge_index"].dropna().astype(int))
        if unmatched:
            p.append(f"{len(unmatched)} edge(s) have no relationship evidence")
        if ev["edge_index"].isna().any():
            p.append("evidence rows that match no graph edge")
        by_edge = ev.dropna(subset=["edge_index"]).groupby("edge_index")["link_type"].agg(lambda s: sorted(set(s)))
        for r in ed.itertuples():
            types = by_edge.get(r.edge_index)
            if types is not None and types != [r.relation]:
                p.append(f"edge {r.edge_index}: relation {r.relation} != evidence {types}")
                break
        # edge flags and e_score re-derived from the evidence and the node data
        ts, hip = nd["ts_s"].to_numpy(), nd["host_ip"].to_numpy(object)
        a, b = ed["local_a"].to_numpy(), ed["local_b"].to_numpy()
        same = ((hip[a] == hip[b]) & (hip[a] != "")).astype(int)
        short = (np.abs(ts[b] - ts[a]) < SHORT_GAP_S).astype(int)
        if not (np.array_equal(same, ed["same_host"].to_numpy()) and np.array_equal(short, ed["short_gap"].to_numpy())):
            p.append("same_host / short_gap flags are not reproducible from the events")
    if len(ev) and len(ed):
        # Stage 4 edge attributes are reproducible from the evidence: e_score = sum(points) + same_host + short_gap,
        # e_log_gap = log1p(min gap). Both are recovered from the model-input values with the saved scaler.
        grp = ev.dropna(subset=["edge_index"]).groupby("edge_index")
        pts, gap = grp["points"].sum(), grp["gap_sec"].min()
        expected = pts.reindex(ed["edge_index"]).to_numpy() + ed["same_host"].to_numpy() + ed["short_gap"].to_numpy()
        if not np.allclose(ed["e_score"].to_numpy(), expected, atol=1e-3):
            p.append("edge e_score is not reproducible from the relationship evidence")
        if not np.allclose(ed["min_gap_sec"].to_numpy(), gap.reindex(ed["edge_index"]).to_numpy(), rtol=1e-3, atol=0.01):
            p.append("edge gap is not reproducible from the relationship evidence")
    return p
