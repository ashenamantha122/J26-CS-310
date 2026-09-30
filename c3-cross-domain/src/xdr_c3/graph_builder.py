"""Stage 4 - graph construction (application version of Stage4_ApproachB_Original_Matching_Logic_Polished).

Reused notebook logic (cell numbers refer to that notebook; formulas, ordering and edge storage are unchanged):
  * cell 8  : node table - events of the graph-eligible scenarios, ordered by (scenario_id, ts_s, kind, row),
              `since_start` / `since_prev`
  * cell 10 : node features - one-hots, log1p numerics, relative-time features, component outputs
  * cell 12 : edges - intra-scenario links grouped per (lo, hi) pair: points = sum, gap = min,
              e_score = points + same_host + short_gap(<300 s), e_log_gap = log1p(gap), one-hot relation columns
  * cell 14 : integrity checks on the finished tables
  * cell 22 : scaling - `(x - mean_train) / std_train` with the SAVED training scaler (never refitted)
  * cell 24 : NPZ packing - per scenario, both edge directions ([lo, hi] block then [hi, lo] block)

Deliberate adaptations for use as an application (each is a stricter or fixed-vocabulary form of the original):
  * one-hot columns and relation columns come from the saved `ModelContract`, not from the data present, so a
    small batch can never shrink or reorder the model input;
  * values outside the saved vocabulary (unknown category, unsupported or ambiguous relation, scenario without an
    edge) are detected and reported, never mapped onto another category/relation. `policy="raise"` stops;
    `policy="quarantine"` removes only the affected scenarios and lists them in the report;
  * no ground truth is read. Evaluation labels are produced separately by `ground_truth_arrays`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .model_contract import EDGE_BASE_FEATURES, EDGE_SCALED, ModelContract, load_contract

MIN_SCENARIO_EVENTS = 2
SHORT_GAP_S = 300            # cell 12: gap_s < 300 -> short_gap flag
FORBIDDEN_NAME_WORDS = ("label", "attack_type", "scenario")   # cell 14: no label-like feature column

# Numeric feature recipe (cell 10): saved name -> (frame, source column, log1p?)
NUMERIC_SPEC = {
    "h_memory_alloc_bytes_log1p": ("host", "memory_alloc_bytes", True),
    "h_disk_read_bytes_log1p": ("host", "disk_read_bytes", True),
    "h_disk_write_bytes_log1p": ("host", "disk_write_bytes", True),
    "h_cpu_usage": ("host", "cpu_usage", False),
    **{f"i_{c}_log1p": ("iot", c, True) for c in
       ["flow_duration", "src_bytes", "dst_bytes", "src_packets", "dst_packets", "iat_mean", "iat_std",
        "packet_size_mean", "packet_size_std", "connection_rate"]},
    "c1_threat_confidence_score": ("host", "threat_confidence_score", False),
    "c2_confidence_score": ("iot", "confidence_score", False),
}
TIME_FEATURES = ("t_since_start_log1p", "t_since_prev_log1p")
CATEGORY_PREFIX = {"h_": "host", "c1_": "host", "i_": "iot", "c2_": "iot"}   # which frame feeds the one-hot


class GraphBuildError(ValueError):
    """The input cannot be turned into model-compatible graphs (see `.report`)."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


@dataclass
class BuildReport:
    unknown_categories: list = field(default_factory=list)    # (feature column, value, node count)
    unsupported_relations: dict = field(default_factory=dict) # link_type -> intra-scenario edges carrying it
    ambiguous_edges: int = 0                                  # edges with more than one relation type
    edgeless_scenarios: list = field(default_factory=list)
    quarantined: dict = field(default_factory=dict)           # scenario_id -> reason(s)

    @property
    def clean(self):
        return not (self.unknown_categories or self.unsupported_relations or self.ambiguous_edges
                    or self.edgeless_scenarios)


@dataclass
class GraphCollection:
    """All graphs, in ascending scenario_id order, in the exact layout of the Stage 4 NPZ files."""
    scenario_ids: np.ndarray        # (G,) int64, ascending
    nodes: pd.DataFrame             # node_id, node, scenario_id, kind, row, ts_s, host_ip  (graph order)
    x: np.ndarray                   # (N, 92) float32, scaled
    node_ptr: np.ndarray            # (G+1,) int64
    lo: np.ndarray                  # (E,) int64 graph-local endpoints of each logical edge
    hi: np.ndarray
    edge_scenario: np.ndarray       # (E,) int64
    es: np.ndarray                  # (E, 14) float32 edge features (one direction)
    edge_ptr_logical: np.ndarray    # (G+1,) int64 logical (undirected) edge offsets
    feature_names: tuple
    feature_groups: tuple
    edge_feature_names: tuple
    report: BuildReport

    @property
    def n_graphs(self):
        return len(self.scenario_ids)

    def arrays(self, scenario_ids=None, node_label=None, scenario_is_attack=None):
        """NPZ-compatible arrays (cell 24) for `scenario_ids` (default: all), in ascending scenario order.
        `node_label` / `scenario_is_attack` are evaluation-only inputs: pass them to obtain the full original
        layout, omit them for inference (the two keys are then absent - nothing is fabricated)."""
        ids = self.scenario_ids if scenario_ids is None else np.sort(np.asarray(scenario_ids, dtype=np.int64))
        missing = ids[~np.isin(ids, self.scenario_ids)]
        if len(missing):
            raise KeyError(f"scenario_ids not in the collection: {missing[:5].tolist()}")
        pos = np.searchsorted(self.scenario_ids, ids)
        nsl = [slice(self.node_ptr[p], self.node_ptr[p + 1]) for p in pos]
        nidx = np.concatenate([np.arange(s.start, s.stop) for s in nsl])
        node_ptr = np.r_[0, np.cumsum([s.stop - s.start for s in nsl])].astype(np.int64)
        ei, ea, eptr = [], [], [0]
        for p in pos:
            sl = slice(self.edge_ptr_logical[p], self.edge_ptr_logical[p + 1])
            lo, hi = self.lo[sl], self.hi[sl]
            ei.append(np.concatenate([np.stack([lo, hi]), np.stack([hi, lo])], axis=1))
            ea.append(np.concatenate([self.es[sl], self.es[sl]], axis=0))
            eptr.append(eptr[-1] + 2 * (sl.stop - sl.start))
        out = {"x": self.x[nidx], "node_ptr": node_ptr,
               "edge_index": np.concatenate(ei, axis=1).astype(np.int64),
               "edge_attr": np.concatenate(ea, axis=0), "edge_ptr": np.array(eptr, dtype=np.int64),
               "scenario_id": ids.astype(np.int64)}
        if node_label is not None:
            out["node_label"] = np.asarray(node_label, dtype=np.int8)[nidx]
        if scenario_is_attack is not None:
            out["scenario_is_attack"] = np.asarray(scenario_is_attack, dtype=np.int8)[pos]
        out["feature_names"] = np.array(self.feature_names, dtype="U80")
        out["feature_groups"] = np.array(self.feature_groups, dtype="U40")
        out["edge_feature_names"] = np.array(self.edge_feature_names, dtype="U40")
        return out


# --------------------------------------------------------------------------- cell 8: nodes
def _node_table(host, iot, assign, summ):
    managed_ips = set(host["host_ip"])
    inc = set(summ["scenario_id"])
    nodes = assign[assign["scenario_id"].isin(inc)].copy()
    nodes["kind"] = nodes["node"].str.split(":").str[0]
    nodes["row"] = nodes["node"].str.split(":").str[1].astype(int)

    h_ts = host["ts"].astype("datetime64[s]").astype("int64").to_numpy()
    i_ts = iot["ts"].astype("datetime64[s]").astype("int64").to_numpy()
    iot_host_ip = np.where(iot["src_ip"].isin(managed_ips), iot["src_ip"],
                           np.where(iot["dst_ip"].isin(managed_ips), iot["dst_ip"], ""))
    kind, rows = nodes["kind"].to_numpy(), nodes["row"].to_numpy()
    isH = kind == "host"
    nodes["ts_s"] = np.where(isH, h_ts[np.where(isH, rows, 0)], i_ts[np.where(~isH, rows, 0)])
    nodes["host_ip"] = np.where(isH, host["host_ip"].to_numpy(object)[np.where(isH, rows, 0)],
                                iot_host_ip[np.where(~isH, rows, 0)])
    nodes = nodes[["node", "scenario_id", "kind", "row", "ts_s", "host_ip"]]
    nodes = nodes.sort_values(["scenario_id", "ts_s", "kind", "row"], kind="stable").reset_index(drop=True)
    nodes["node_id"] = np.arange(len(nodes))
    g = nodes.groupby("scenario_id")["ts_s"]
    nodes["since_start"] = nodes["ts_s"] - g.transform("min")
    nodes["since_prev"] = g.diff().fillna(0)
    return nodes


# --------------------------------------------------------------------------- cell 10: features (fixed vocabulary)
def _parse_category(name):
    """'h_event_type=file_event' -> ('h_', 'event_type', 'file_event')."""
    left, label = name.split("=", 1)
    for prefix in CATEGORY_PREFIX:
        if left.startswith(prefix):
            return prefix, left[len(prefix):], label
    raise GraphBuildError(f"unrecognised categorical feature in saved metadata: {name}")


def _node_features(nodes, host, iot, contract, report):
    n = len(nodes)
    kind, rows = nodes["kind"].to_numpy(), nodes["row"].to_numpy()
    frames = {"host": host, "iot": iot}
    kind_mask = {"host": kind == "host", "iot": kind == "iot"}
    feat, observed = {}, {}

    def values(frame_name, col):
        v = np.full(n, None, dtype=object)
        m = kind_mask[frame_name]
        v[m] = frames[frame_name][col].to_numpy(object)[rows[m]]
        return v

    vocab = {}                     # (frame, column) -> labels in the saved vocabulary
    for name in contract.feature_names:
        if "=" in name:
            prefix, col, label = _parse_category(name)
            vocab.setdefault((CATEGORY_PREFIX[prefix], col, prefix), []).append(label)

    for name, group in zip(contract.feature_names, contract.feature_groups):
        if name in ("is_host", "is_iot"):
            feat[name] = (kind == name[3:]).astype(np.int8)
        elif "=" in name:
            prefix, col, label = _parse_category(name)
            fr = CATEGORY_PREFIX[prefix]
            vals = observed.setdefault((fr, col, prefix), values(fr, col))
            na = np.isin(vals, ["-", ""]) if label == "none" else np.zeros(n, dtype=bool)
            feat[name] = ((vals == label) | na).astype(np.int8)
        elif name in NUMERIC_SPEC:
            fr, col, log = NUMERIC_SPEC[name]
            x = np.zeros(n)
            m = kind_mask[fr]
            x[m] = pd.to_numeric(frames[fr][col]).to_numpy(float)[rows[m]]
            if log and x.min() < 0:
                raise GraphBuildError(f"negative value in log1p feature {name}")
            feat[name] = np.log1p(x) if log else x
        elif name == "t_since_start_log1p":
            feat[name] = np.log1p(nodes["since_start"].to_numpy(float))
        elif name == "t_since_prev_log1p":
            feat[name] = np.log1p(nodes["since_prev"].to_numpy(float))
        else:
            raise GraphBuildError(f"saved feature has no construction recipe: {name}")

    # values outside the saved one-hot vocabulary would silently become all-zero rows: detect them
    for (fr, col, prefix), vals in observed.items():
        known = set(vocab[(fr, col, prefix)])
        has_none = "none" in known
        present = pd.Series(vals[kind_mask[fr]]).value_counts()
        for v, cnt in present.items():
            ok = v in known or (has_none and v in ("-", ""))
            if not ok:
                report.unknown_categories.append((f"{prefix}{col}", v, int(cnt)))
    return pd.DataFrame(feat, columns=list(contract.feature_names))


# --------------------------------------------------------------------------- cell 12: edges (fixed relations)
def _edges(nodes, links, contract, report):
    nid = pd.Series(nodes["node_id"].to_numpy(), index=nodes["node"].to_numpy())
    scn = pd.Series(nodes["scenario_id"].to_numpy(), index=nodes["node"].to_numpy())
    l = links[links["node_a"].isin(nid.index) & links["node_b"].isin(nid.index)].copy()
    l["a"], l["b"] = nid.reindex(l["node_a"]).to_numpy(), nid.reindex(l["node_b"]).to_numpy()
    l["sa"], l["sb"] = scn.reindex(l["node_a"]).to_numpy(), scn.reindex(l["node_b"]).to_numpy()
    l = l[l["sa"] == l["sb"]].copy()
    l["lo"], l["hi"] = np.minimum(l["a"], l["b"]), np.maximum(l["a"], l["b"])

    pair = l.groupby(["lo", "hi"]).agg(scenario_id=("sa", "first"), points=("points", "sum"), gap=("gap_sec", "min"))
    types = pd.crosstab([l["lo"], l["hi"]], l["link_type"]).gt(0).astype(int)
    supported = list(contract.relation_link_types)
    extra = [c for c in types.columns if c not in supported]           # link types the model was never trained on
    for c in extra:
        report.unsupported_relations[c] = int(types[c].sum())
    unsupported_pairs = types[extra].any(axis=1) if extra else pd.Series(False, index=types.index)
    types = types.reindex(columns=supported, fill_value=0)
    ambiguous_pairs = types.sum(axis=1) > 1                            # several relations on one edge: argmax would guess
    report.ambiguous_edges = int(ambiguous_pairs.sum())
    types.columns = ["e_type=" + c for c in types.columns]
    edges = pair.join(types)
    edges["bad_relation"] = unsupported_pairs.reindex(edges.index).to_numpy() | ambiguous_pairs.reindex(edges.index).to_numpy()
    edges = edges.reset_index().sort_values(["scenario_id", "lo", "hi"], kind="stable").reset_index(drop=True)

    ts_n, hip_n = nodes["ts_s"].to_numpy(), nodes["host_ip"].to_numpy(object)
    gap_s = np.abs(ts_n[edges["hi"]] - ts_n[edges["lo"]])
    same_host = ((hip_n[edges["lo"]] == hip_n[edges["hi"]]) & (hip_n[edges["lo"]] != "")).astype(int)
    short_gap = (gap_s < SHORT_GAP_S).astype(int)
    edges["e_score"] = edges["points"] + same_host + short_gap
    edges["e_log_gap"] = np.log1p(edges["gap"])
    edges["e_same_host"], edges["e_short_gap"] = same_host, short_gap
    return edges


# --------------------------------------------------------------------------- orchestration
def build_graphs(host, iot, assign, summ, links, contract: ModelContract = None, policy="raise") -> GraphCollection:
    """Build the graph collection from Stage 2 inference frames and the Stage 3b hand-off tables
    (`Stage3bResult.stage4_inputs()`: assign, summ, links). `host` / `iot` must not carry ground truth."""
    if policy not in ("raise", "quarantine"):
        raise ValueError("policy must be 'raise' or 'quarantine'")
    contract = contract or load_contract()
    from .stage2_validate import assert_no_ground_truth
    assert_no_ground_truth(host)
    assert_no_ground_truth(iot)
    report = BuildReport()

    nodes = _node_table(host, iot, assign, summ)
    sizes = nodes.groupby("scenario_id").size()
    if not (sizes >= MIN_SCENARIO_EVENTS).all():
        raise GraphBuildError("a scenario has fewer than 2 nodes")
    if not (sizes.reindex(summ["scenario_id"]).to_numpy() == summ["n_events"].to_numpy()).all():
        raise GraphBuildError("scenario sizes differ from the Stage 3b summary")
    X = _node_features(nodes, host, iot, contract, report)
    edges = _edges(nodes, links, contract, report)

    # --- scenarios that cannot be represented in the model's input space
    bad = {}
    node_bad = np.zeros(len(nodes), dtype=bool)
    for (prefix_col, value, _cnt) in report.unknown_categories:
        prefix = next(p for p in CATEGORY_PREFIX if prefix_col.startswith(p))
        fr, col = CATEGORY_PREFIX[prefix], prefix_col[len(prefix):]
        src = {"host": host, "iot": iot}[fr][col].to_numpy(object)
        m = (nodes["kind"].to_numpy() == fr) & (src[nodes["row"].to_numpy()] == value)
        node_bad |= m
    for sid in nodes.loc[node_bad, "scenario_id"].unique():
        bad.setdefault(int(sid), []).append("unknown category value")
    for sid in edges.loc[edges["bad_relation"], "scenario_id"].unique():
        bad.setdefault(int(sid), []).append("unsupported or ambiguous relation")
    with_edge = set(edges["scenario_id"].unique())
    for sid in sorted(set(nodes["scenario_id"].unique()) - with_edge):
        report.edgeless_scenarios.append(int(sid))
        bad.setdefault(int(sid), []).append("scenario has no edge")
    report.quarantined = bad
    if bad and policy == "raise":
        raise GraphBuildError(
            f"{len(bad)} scenario(s) cannot be represented in the saved model contract: "
            f"unknown categories={report.unknown_categories[:5]}, unsupported relations={report.unsupported_relations}, "
            f"ambiguous edges={report.ambiguous_edges}, edgeless scenarios={report.edgeless_scenarios[:5]}", report)
    if bad:                                           # quarantine: drop only the affected scenarios
        keep_nodes = ~nodes["scenario_id"].isin(bad).to_numpy()
        nodes, X = nodes[keep_nodes].reset_index(drop=True), X[keep_nodes].reset_index(drop=True)
        edges = edges[~edges["scenario_id"].isin(bad)].reset_index(drop=True)
        old_to_new = np.full(keep_nodes.size, -1, dtype=np.int64)
        old_to_new[np.flatnonzero(keep_nodes)] = np.arange(keep_nodes.sum())
        edges["lo"], edges["hi"] = old_to_new[edges["lo"]], old_to_new[edges["hi"]]
        nodes["node_id"] = np.arange(len(nodes))

    # --- cell 14 integrity checks
    if not (np.isfinite(X.to_numpy(float)).all()):
        raise GraphBuildError("NaN / inf in node features")
    if any(w in c.lower() for c in X.columns for w in FORBIDDEN_NAME_WORDS):
        raise GraphBuildError("label-like column among the features")
    sc_of_node = nodes["scenario_id"].to_numpy()
    if not ((sc_of_node[edges["lo"]] == edges["scenario_id"].to_numpy()).all()
            and (sc_of_node[edges["hi"]] == edges["scenario_id"].to_numpy()).all()):
        raise GraphBuildError("an edge crosses scenarios")
    rel = edges[[f for f in contract.relation_names]].to_numpy()
    if not (rel.sum(axis=1) == 1).all():
        raise GraphBuildError("relation columns are not exactly one-hot")

    # --- cell 22: SAVED scaler, applied in float64 exactly as the original did
    Xs = X.astype(np.float64)
    for c, (mu, sd) in contract.node_scaler.items():
        Xs[c] = (X[c] - mu) / sd
    Es = edges[list(contract.edge_feature_names)].astype(np.float64)
    for c, (mu, sd) in contract.edge_scaler.items():
        Es[c] = (edges[c] - mu) / sd

    # --- graph boundaries (cell 24 inputs)
    sids = np.sort(nodes["scenario_id"].unique()).astype(np.int64)
    ncount = nodes.groupby("scenario_id").size().reindex(sids).to_numpy()
    ecount = edges.groupby("scenario_id").size().reindex(sids).to_numpy()
    node_ptr = np.r_[0, np.cumsum(ncount)].astype(np.int64)
    edge_ptr = np.r_[0, np.cumsum(ecount)].astype(np.int64)
    nstart_of = np.repeat(node_ptr[:-1], ecount)                  # edges are sorted by scenario, so this lines up
    nodes_out = nodes[["node_id", "node", "scenario_id", "kind", "row", "ts_s", "host_ip"]].copy()
    return GraphCollection(
        scenario_ids=sids, nodes=nodes_out, x=Xs.to_numpy(np.float32), node_ptr=node_ptr,
        lo=(edges["lo"].to_numpy() - nstart_of).astype(np.int64), hi=(edges["hi"].to_numpy() - nstart_of).astype(np.int64),
        edge_scenario=edges["scenario_id"].to_numpy(np.int64), es=Es.to_numpy(np.float32), edge_ptr_logical=edge_ptr,
        feature_names=tuple(contract.feature_names), feature_groups=tuple(contract.feature_groups),
        edge_feature_names=tuple(contract.edge_feature_names), report=report)


def build_graphs_from_stage3b(data, stage3b, contract: ModelContract = None, policy="raise") -> GraphCollection:
    """Stage 2 `Stage2Data` + `Stage3bResult` -> graphs, through `stage3b.stage4_inputs()`."""
    host, iot = data.to_stage3_frames()           # inference frames: no ground truth
    s4 = stage3b.stage4_inputs()
    return build_graphs(host, iot, s4["assign"], s4["summ"], s4["links"], contract, policy)


# --------------------------------------------------------------------------- evaluation only
def ground_truth_arrays(collection: GraphCollection, data):
    """(node_label int8 per node, scenario_is_attack int8 per graph) in the collection's order.
    Offline evaluation / NPZ comparison only: never an input of graph construction."""
    h = data.host_truth["label"].astype(int).to_numpy()
    i = data.iot_truth["label"].astype(int).to_numpy()
    nd = collection.nodes
    node_label = np.where(nd["kind"].to_numpy() == "host", h[np.where(nd["kind"] == "host", nd["row"], 0)],
                          i[np.where(nd["kind"] == "iot", nd["row"], 0)]).astype(np.int8)
    attack = (pd.Series(node_label).groupby(nd["scenario_id"].to_numpy()).max()
              .reindex(collection.scenario_ids).to_numpy() > 0).astype(np.int8)
    return node_label, attack
