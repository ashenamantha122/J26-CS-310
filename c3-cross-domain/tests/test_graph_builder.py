"""Tests for xdr_c3.graph_builder / model_contract (Stage 4) on the real data, against the saved Stage 4 outputs.

Negative tests mutate COPIES of real tables (never the files); no dummy or invented telemetry is used."""
import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xdr_c3 import graph_builder as gb
from xdr_c3 import scenario_grouping as sg
from xdr_c3 import stage2_validate as s2
from xdr_c3.config import (CHECKPOINT_PATH, FEATURE_NAMES_CSV, LOCKED_SELECTION_JSON, MODEL_CONFIG_JSON,
                           SCALER_CSV, SPLIT_NPZ, STAGE4_OUT)
from xdr_c3.model_contract import ContractError, load_contract, read_checkpoint_config

GD = STAGE4_OUT / "Graph Data"
NPZ_KEYS = {"x", "node_ptr", "edge_index", "edge_attr", "edge_ptr", "scenario_id", "scenario_is_attack",
            "node_label", "feature_names", "feature_groups", "edge_feature_names"}


@pytest.fixture(scope="session")
def truth(graphs, s2_data):
    return gb.ground_truth_arrays(graphs, s2_data)       # evaluation only


@pytest.fixture(scope="session")
def npz():
    return {k: dict(np.load(p)) for k, p in SPLIT_NPZ.items()}


def build(s2_data, s3b, contract, summ=None, links=None, host=None, policy="raise"):
    h, i = s2_data.to_stage3_frames()
    s4 = s3b.stage4_inputs()
    return gb.build_graphs(host if host is not None else h, i, s4["assign"], s4["summ"] if summ is None else summ,
                           s4["links"] if links is None else links, contract, policy)


# =========================================================================== 1. Stage 3b hand-off
def test_stage3b_handoff_builds_graphs(graphs, s3b):
    s4 = s3b.stage4_inputs()
    assert set(s4) == {"assign", "summ", "links"}
    assert graphs.n_graphs == 12780 == len(s4["summ"])
    assert len(graphs.nodes) == 101695
    assert graphs.report.clean and not graphs.report.quarantined


# =========================================================================== 2. scenario membership / 3. ordering
def test_scenario_ids_and_membership_match_saved_stage4(graphs, s3b):
    assert np.array_equal(graphs.scenario_ids, s3b.summary["scenario_id"].to_numpy())
    nl = pd.read_csv(GD / "final_node_labels_v6v2 (1).csv", usecols=["node_id", "node", "scenario_id"])
    assert nl["node_id"].equals(graphs.nodes["node_id"])
    assert nl["node"].equals(graphs.nodes["node"])
    assert nl["scenario_id"].equals(graphs.nodes["scenario_id"])
    per_scn = graphs.nodes.groupby("scenario_id").size().reindex(graphs.scenario_ids).to_numpy()
    assert np.array_equal(per_scn, np.diff(graphs.node_ptr))
    assert (s3b.summary["n_events"].to_numpy() == per_scn).all()


def test_original_source_row_identities_preserved(graphs, s2_data):
    nd = graphs.nodes
    assert (nd["node"] == nd["kind"] + ":" + nd["row"].astype(str)).all()
    assert nd["row"][nd["kind"] == "host"].max() < len(s2_data.host)
    assert nd["row"][nd["kind"] == "iot"].max() < len(s2_data.iot)
    assert nd["node"].is_unique


def test_node_ordering_is_deterministic_and_time_ordered(graphs, s2_data, s3b, contract):
    nd = graphs.nodes
    key = nd[["scenario_id", "ts_s", "kind", "row"]].to_numpy(dtype=object)
    order = np.lexsort((nd["row"], nd["kind"], nd["ts_s"], nd["scenario_id"]))
    assert np.array_equal(order, np.arange(len(nd)))
    again = build(s2_data, s3b, contract)
    assert again.nodes.equals(graphs.nodes) and np.array_equal(again.x, graphs.x)
    assert key.shape[0] == len(nd)


# =========================================================================== 4. node features / 7. scaler / 8. vocabulary
def test_node_feature_values_and_order_equal_saved_npz(graphs, truth, npz):
    for name, ref in npz.items():
        a = graphs.arrays(ref["scenario_id"], *truth)
        assert a["x"].dtype == ref["x"].dtype == np.float32
        assert np.array_equal(a["x"], ref["x"])                     # exact: tolerance 0
        assert np.array_equal(a["node_ptr"], ref["node_ptr"])
        assert np.array_equal(a["node_label"], ref["node_label"])


def test_feature_order_matches_saved_metadata_and_checkpoint(graphs, contract):
    fn = pd.read_csv(FEATURE_NAMES_CSV)
    assert list(graphs.feature_names) == list(fn["feature"]) == list(contract.feature_names)
    assert list(graphs.feature_groups) == list(fn["group"])
    assert len(graphs.feature_names) == 92 and len(contract.model_cols) == 84
    ck = read_checkpoint_config()
    assert list(contract.model_cols) == ck["cols"] == [i for i, g in enumerate(fn["group"]) if g != "component"]
    assert ck["epoch"] == 40 and json.loads(LOCKED_SELECTION_JSON.read_text())["selected_epoch"] == 40
    assert json.loads(MODEL_CONFIG_JSON.read_text())["input_features"] == 84


def test_saved_scaler_is_applied_not_refitted(graphs, contract):
    sc = pd.read_csv(SCALER_CSV)
    assert {c: (m, s) for c, m, s in zip(sc[sc.table == "node"].column, sc[sc.table == "node"].mean_train,
                                          sc[sc.table == "node"].std_train)} == contract.node_scaler
    names = list(graphs.feature_names)
    unscaled_cols = [i for i, n in enumerate(names) if n not in contract.node_scaler]
    # unscaled indicator columns are untouched 0/1
    assert set(np.unique(graphs.x[:, unscaled_cols])) <= {0.0, 1.0}
    # undoing the saved scaler reproduces the CSV values written by the notebook (6-decimal rounding -> 1e-5)
    raw = pd.read_csv(GD / "final_node_features_v6v2.csv")
    for c, (mu, sd) in contract.node_scaler.items():
        back = graphs.x[:, names.index(c)].astype(np.float64) * sd + mu
        assert np.abs(back - raw[c].to_numpy()).max() < 1e-5 * max(sd, 1.0) + 1e-5
    # training nodes are ~N(0,1) under the training scaler; the whole-data statistics are not (it was not refitted)
    train_ids = np.load(SPLIT_NPZ["train"])["scenario_id"]
    in_train = graphs.nodes["scenario_id"].isin(train_ids).to_numpy()
    j = names.index("t_since_start_log1p")
    assert abs(graphs.x[in_train, j].mean()) < 1e-4 and abs(graphs.x[in_train, j].std() - 1) < 1e-4
    assert abs(graphs.x[~in_train, j].mean()) > 1e-4


def test_one_hot_vocabulary_is_fixed_not_batch_derived(s2_data, s3b, contract, graphs):
    ids = graphs.scenario_ids[[100, 5000, 12000]]
    summ = s3b.summary[s3b.summary["scenario_id"].isin(ids)]
    small = build(s2_data, s3b, contract, summ=summ)
    assert list(small.feature_names) == list(contract.feature_names) and small.x.shape[1] == 92
    assert small.edge_feature_names == graphs.edge_feature_names
    ref = graphs.arrays(ids)
    got = small.arrays()
    assert np.array_equal(got["x"], ref["x"]) and np.array_equal(got["edge_attr"], ref["edge_attr"])
    assert np.array_equal(got["edge_index"], ref["edge_index"])
    # a tiny batch uses far fewer categories than the vocabulary, yet no column disappears
    onehot = [i for i, n in enumerate(contract.feature_names) if "=" in n]
    assert (small.x[:, onehot].sum(axis=0) == 0).any()


# =========================================================================== 5. edges / 6. bidirectional / 9. relations
def test_edge_features_and_order_equal_saved_npz(graphs, truth, npz):
    for ref in npz.values():
        a = graphs.arrays(ref["scenario_id"], *truth)
        assert a["edge_attr"].dtype == np.float32
        assert np.array_equal(a["edge_attr"], ref["edge_attr"])
        assert np.array_equal(a["edge_index"], ref["edge_index"])
        assert np.array_equal(a["edge_ptr"], ref["edge_ptr"])


def test_edges_are_stored_in_both_directions(graphs):
    a = graphs.arrays()
    ei, ea, ptr = a["edge_index"], a["edge_attr"], a["edge_ptr"]
    for g in np.random.default_rng(0).choice(graphs.n_graphs, 300, replace=False):
        s, e = ptr[g], ptr[g + 1]
        half = (e - s) // 2
        assert (e - s) % 2 == 0 and half >= 1
        fwd, rev = ei[:, s:s + half], ei[:, s + half:e]
        assert np.array_equal(fwd, rev[::-1]) and (fwd[0] < fwd[1]).all()      # lo < hi forward, mirrored back
        assert np.array_equal(ea[s:s + half], ea[s + half:e])
    assert ei.shape[1] == 2 * len(graphs.es) == 184308


def test_relation_vocabulary_is_fixed_and_one_hot(graphs, contract):
    assert contract.relation_names == tuple(json.loads(MODEL_CONFIG_JSON.read_text())["relations"])
    assert len(contract.relation_names) == 10 and contract.relation_cols == tuple(range(4, 14))
    assert graphs.edge_feature_names[4:] == contract.relation_names
    rel = graphs.es[:, 4:14]
    assert set(np.unique(rel)) == {0.0, 1.0} and (rel.sum(axis=1) == 1).all()
    assert (rel.sum(axis=0) > 0).sum() == 10                      # all ten trained relations occur
    assert "e_type=flow_same_src" not in graphs.edge_feature_names


def test_edge_scaling_uses_saved_scaler_on_two_columns(graphs, contract):
    fe = pd.read_csv(GD / "final_edges_v6v2.csv")
    for c, (mu, sd) in contract.edge_scaler.items():
        j = graphs.edge_feature_names.index(c)
        back = graphs.es[:, j].astype(np.float64) * sd + mu
        assert np.abs(back - fe[c].to_numpy()).max() < 1e-4
    assert tuple(contract.edge_scaler) == ("e_score", "e_log_gap")
    offs = np.repeat(graphs.node_ptr[:-1], np.diff(graphs.edge_ptr_logical))
    assert np.array_equal(fe["node_a_id"], graphs.lo + offs) and np.array_equal(fe["node_b_id"], graphs.hi + offs)
    assert len(fe) == len(graphs.es) == 92154


# =========================================================================== 10. ground-truth independence
def test_builder_code_never_references_ground_truth():
    src = Path(gb.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"label", "type", "attack_type", "predicted_class", "node_label", "scenario_is_attack", "n_label1"}
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "ground_truth_arrays"]:
        doc = ast.get_docstring(fn, clean=False)
        for n in ast.walk(fn):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value != doc:
                assert n.value not in banned, (fn.name, n.value)
    params = {a.arg for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name.startswith("build_graphs")
              for a in fn.args.args}
    assert not params & {"labels", "label", "truth", "node_label", "is_attack"}


def test_builder_rejects_frames_that_carry_ground_truth(s2_data, s3b, contract):
    host, iot = s2_data.to_stage3_frames(include_ground_truth=True)
    s4 = s3b.stage4_inputs()
    with pytest.raises(ValueError):
        gb.build_graphs(host, iot, s4["assign"], s4["summ"], s4["links"], contract)


def test_label_free_pipeline_gives_identical_model_input(s2_data, graphs, npz):
    """Remove every ground-truth column, rerun Stage 2 -> 3b -> 4, and compare to the original NPZ files."""
    h = s2_data.host.drop(columns=["ts", "row"]).astype({"process_id": str, "parent_process_id": str})
    i = s2_data.iot.drop(columns=["ts", "row"])
    unlabelled = s2.validate_and_build(h, i, allow_missing_ground_truth=True)
    r2 = sg.run_stage3b(unlabelled)
    g2 = gb.build_graphs_from_stage3b(unlabelled, r2)
    assert np.array_equal(g2.scenario_ids, graphs.scenario_ids) and np.array_equal(g2.x, graphs.x)
    for name, ref in npz.items():
        a = g2.arrays(ref["scenario_id"])                                   # no labels supplied
        assert set(a) == NPZ_KEYS - {"node_label", "scenario_is_attack"}
        for k in ("x", "node_ptr", "edge_index", "edge_attr", "edge_ptr", "scenario_id"):
            assert np.array_equal(a[k], ref[k]), (name, k)


def test_shuffled_ground_truth_does_not_change_graphs(s2_data, graphs):
    """Ground truth lives only in the separate truth frames; shuffling it cannot touch the model input."""
    rng = np.random.default_rng(5)
    shuffled = s2.Stage2Data(s2_data.host, s2_data.iot, s2_data.host_truth.sample(frac=1, random_state=1).set_axis(
        s2_data.host_truth.index), s2_data.iot_truth.iloc[rng.permutation(len(s2_data.iot_truth))].set_axis(
        s2_data.iot_truth.index), s2_data.report)
    nl_a, _ = gb.ground_truth_arrays(graphs, s2_data)
    nl_b, _ = gb.ground_truth_arrays(graphs, shuffled)
    assert (nl_a != nl_b).any()                                       # evaluation labels did change ...
    assert all(f"{c}" not in graphs.feature_names for c in ("label", "type", "attack_type"))   # ... model input did not


# =========================================================================== 11. unsupported / ambiguous relations, unknowns
def _single_link_pair_in_scenario(s3b):
    links, sc = s3b.links, dict(zip(s3b.assignments["node"], s3b.assignments["scenario_id"]))
    summ_ids = set(s3b.summary["scenario_id"])
    pair_n = links.groupby(["node_a", "node_b"]).size()
    for idx, r in links.iterrows():
        if r.link_type == "same_process" and pair_n[(r.node_a, r.node_b)] == 1 and sc[r.node_a] == sc[r.node_b] \
                and sc[r.node_a] in summ_ids:
            return idx, sc[r.node_a]
    raise AssertionError("no suitable link")


def test_unsupported_relation_is_detected_not_mapped(s2_data, s3b, contract, graphs):
    idx, sid = _single_link_pair_in_scenario(s3b)
    links = s3b.links.copy()
    links.loc[idx, "link_type"] = "flow_same_src"          # a real Stage 3 link type the model was never trained on
    with pytest.raises(gb.GraphBuildError) as e:
        build(s2_data, s3b, contract, links=links)
    assert e.value.report.unsupported_relations == {"flow_same_src": 1}
    q = build(s2_data, s3b, contract, links=links, policy="quarantine")
    assert list(q.report.quarantined) == [sid] and "unsupported or ambiguous relation" in q.report.quarantined[sid][0]
    assert q.n_graphs == graphs.n_graphs - 1 and sid not in q.scenario_ids
    keep = graphs.scenario_ids[graphs.scenario_ids != sid]            # all other graphs are byte-for-byte unchanged
    for k in ("x", "edge_index", "edge_attr", "node_ptr", "edge_ptr"):
        assert np.array_equal(q.arrays()[k], graphs.arrays(keep)[k])


def test_ambiguous_multi_relation_edge_is_detected(s2_data, s3b, contract):
    idx, sid = _single_link_pair_in_scenario(s3b)
    dup = s3b.links.loc[[idx]].copy()
    dup["link_type"] = "parent_child"                    # same pair now carries two supported relations
    links = pd.concat([s3b.links, dup], ignore_index=True)
    with pytest.raises(gb.GraphBuildError) as e:
        build(s2_data, s3b, contract, links=links)
    assert e.value.report.ambiguous_edges == 1
    assert list(build(s2_data, s3b, contract, links=links, policy="quarantine").report.quarantined) == [sid]


def test_unknown_category_value_is_detected(s2_data, s3b, contract):
    host, _ = s2_data.to_stage3_frames()
    row = int(s3b.assignments.loc[s3b.assignments["node"].str.startswith("host:"), "node"].iloc[1000].split(":")[1])
    host = host.copy()
    host.loc[host["row"] == row, "process_name"] = "unseen-process"
    with pytest.raises(gb.GraphBuildError) as e:
        build(s2_data, s3b, contract, host=host)
    assert ("h_process_name", "unseen-process", 1) in e.value.report.unknown_categories


def test_scenario_without_edges_is_reported(s2_data, s3b, contract):
    _, sid = _single_link_pair_in_scenario(s3b)
    sc = dict(zip(s3b.assignments["node"], s3b.assignments["scenario_id"]))
    in_s = s3b.links["node_a"].map(sc).eq(sid) | s3b.links["node_b"].map(sc).eq(sid)
    links = s3b.links[~in_s]
    with pytest.raises(gb.GraphBuildError) as e:
        build(s2_data, s3b, contract, links=links)
    assert sid in e.value.report.edgeless_scenarios
    assert sid not in build(s2_data, s3b, contract, links=links, policy="quarantine").scenario_ids


# =========================================================================== 12. NPZ contract
def test_npz_contract_compatibility(graphs, truth, npz):
    a = graphs.arrays(node_label=truth[0], scenario_is_attack=truth[1])
    assert set(a) == NPZ_KEYS == set(np.load(SPLIT_NPZ["train"]).files)
    ref = npz["train"]
    for k in a:
        assert a[k].dtype == ref[k].dtype, k
    assert a["x"].shape[1] == 92 and a["edge_attr"].shape[1] == 14
    assert a["node_ptr"][-1] == len(a["x"]) and a["edge_ptr"][-1] == len(a["edge_attr"]) == a["edge_index"].shape[1]
    assert np.isfinite(a["x"]).all() and np.isfinite(a["edge_attr"]).all()
    assert np.array_equal(a["scenario_id"], np.sort(a["scenario_id"]))
    for j in np.random.default_rng(1).choice(graphs.n_graphs, 500, replace=False):
        s, e = a["edge_ptr"][j], a["edge_ptr"][j + 1]
        assert a["edge_index"][:, s:e].max() < a["node_ptr"][j + 1] - a["node_ptr"][j]


def test_inference_arrays_omit_evaluation_labels(graphs):
    a = graphs.arrays()
    assert "node_label" not in a and "scenario_is_attack" not in a


# =========================================================================== 13. failure cases
def test_invalid_policy_and_unknown_scenarios(s2_data, s3b, contract, graphs):
    with pytest.raises(ValueError):
        build(s2_data, s3b, contract, policy="ignore")
    with pytest.raises(KeyError):
        graphs.arrays([10 ** 9])


def test_summary_size_mismatch_is_rejected(s2_data, s3b, contract):
    summ = s3b.summary.copy()
    summ.loc[summ.index[0], "n_events"] += 1
    with pytest.raises(gb.GraphBuildError, match="sizes differ"):
        build(s2_data, s3b, contract, summ=summ)


def test_negative_log_feature_is_rejected(s2_data, s3b, contract):
    host, _ = s2_data.to_stage3_frames()
    host = host.copy()
    row = int(s3b.assignments.loc[s3b.assignments["node"].str.startswith("host:"), "node"].iloc[10].split(":")[1])
    host.loc[host["row"] == row, "memory_alloc_bytes"] = "-5"
    with pytest.raises(gb.GraphBuildError, match="negative"):
        build(s2_data, s3b, contract, host=host)


def test_inconsistent_saved_metadata_is_rejected(tmp_path):
    bad = tmp_path / "scaler.csv"
    sc = pd.read_csv(SCALER_CSV)
    sc[sc.column != "e_log_gap"].to_csv(bad, index=False)
    with pytest.raises(ContractError):
        load_contract(scaler_csv=bad)
    names = tmp_path / "names.csv"
    fn = pd.read_csv(FEATURE_NAMES_CSV)
    fn.iloc[:-1].to_csv(names, index=False)
    with pytest.raises(ContractError):
        load_contract(feature_names_csv=names)


def test_checkpoint_config_is_read_without_modifying_it():
    before = CHECKPOINT_PATH.stat().st_mtime_ns, CHECKPOINT_PATH.stat().st_size
    cfg = read_checkpoint_config()
    assert (CHECKPOINT_PATH.stat().st_mtime_ns, CHECKPOINT_PATH.stat().st_size) == before
    assert cfg["kind"] == "rgcn" and cfg["f_in"] == 84 and cfg["e_dim"] == 14 and len(cfg["rel_names"]) == 10


# =========================================================================== 14. real-data reference equivalence
def test_real_data_graph_collection_equals_all_four_saved_npz_files(graphs, truth, npz):
    expected = {"train": 8946, "validation": 1278, "test_normal": 1112, "test_attack": 1444}
    seen = []
    for name, ref in npz.items():
        assert len(ref["scenario_id"]) == expected[name]
        a = graphs.arrays(ref["scenario_id"], *truth)          # reconciled by scenario id, not by position
        assert set(a) == set(ref)
        for k, v in ref.items():
            if v.dtype.kind in "US":
                assert (a[k] == v).all(), (name, k)
            else:
                assert a[k].dtype == v.dtype and np.array_equal(a[k], v), (name, k)    # tolerance 0
        seen.extend(ref["scenario_id"].tolist())
    assert sorted(seen) == graphs.scenario_ids.tolist() and len(set(seen)) == 12780
    assert len(graphs.nodes) == 101695 and len(graphs.es) == 92154 and 2 * len(graphs.es) == 184308
