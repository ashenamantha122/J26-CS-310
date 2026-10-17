"""Tests for xdr_c3.scenario_grouping (Stage 3b) on the real Host / IoT data, against the saved Stage 3b references."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xdr_c3 import relationship_mapping as rm
from xdr_c3 import scenario_grouping as sg
from xdr_c3 import stage2_validate as s2
from xdr_c3.config import PROJECT_ROOT

STAGES = PROJECT_ROOT / "Stages"
NB3B = STAGES / "Stage 3b" / "Stage3b_Scenario_Grouping_V6v2.ipynb"
REF = STAGES / "Stage 3b" / "Outputs"
MARKER = "# =========================================================================== application integration"


@pytest.fixture(scope="module")
def data():
    return s2.load_and_validate()


@pytest.fixture(scope="module")
def result(data):
    return sg.run_stage3b(data)


@pytest.fixture(scope="module")
def evaluation(result, data):
    return sg.evaluation_tables(result, data)


@pytest.fixture(scope="module")
def ref_links():
    return pd.read_csv(REF / "links_stage3b_v6v2.csv")


@pytest.fixture(scope="module")
def ref_assign():
    return pd.read_csv(REF / "scenario_assignments_v6v2.csv")


@pytest.fixture(scope="module")
def ref_summary():
    return pd.read_csv(REF / "scenario_summary_v6v2.csv", keep_default_na=False)   # empty `hosts` stays ""


# --------------------------------------------------------------------------- extraction is verbatim
def test_module_is_verbatim_notebook_code():
    nb = json.loads(NB3B.read_text(encoding="utf-8"))
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "".join(c["source"]).startswith("%%writefile scenario_grouping.py"))
    original = cell.split("\n", 1)[1].rstrip("\n")
    assert Path(sg.__file__).read_text(encoding="utf-8").split(MARKER)[0].rstrip("\n") == original


def test_original_routines_and_constants_present():
    for name in ("node_table", "node_index", "pair_scores", "group_scenarios", "scenario_table",
                 "handoff_links", "remote_source_links"):
        assert callable(getattr(sg, name))
    assert (sg.GATE_GAP, sg.SHORT_GAP, sg.T_FINAL, sg.MIN_SCENARIO_EVENTS) == (900, 300, 3, 2)


# --------------------------------------------------------------------------- combined links
def test_combined_links_match_reference_exactly(result, ref_links, tmp_path):
    assert len(result.links) == len(ref_links) == 99095
    assert result.links.link_type.value_counts().to_dict() == ref_links.link_type.value_counts().to_dict()
    assert result.links.link_type.value_counts().to_dict()["service_handoff"] == 1219
    assert result.links.link_type.value_counts().to_dict()["same_remote_source"] == 6413
    pd.testing.assert_frame_equal(result.links.reset_index(drop=True), ref_links, check_exact=True)
    out = tmp_path / "l.csv"
    result.links.to_csv(out, index=False, lineterminator="\n")
    assert out.read_bytes() == (REF / "links_stage3b_v6v2.csv").read_bytes()


def test_stage3_links_are_a_prefix_of_combined_links(data, result):
    stage3 = rm.build_links_from_stage2(data)
    pd.testing.assert_frame_equal(result.links.iloc[:len(stage3)].reset_index(drop=True), stage3, check_exact=True)


# --------------------------------------------------------------------------- scenario assignments and summary
def test_assignments_exactly_match_reference(result, evaluation, ref_assign):
    pd.testing.assert_frame_equal(result.assignments, ref_assign[sg.ASSIGNMENT_COLUMNS], check_exact=True)
    pd.testing.assert_frame_equal(evaluation[0], ref_assign, check_exact=True)       # incl. label-derived `type`
    assert np.array_equal(result.scenario, ref_assign["scenario_id"].to_numpy())     # ids deterministic, no remapping


def test_summary_structural_columns_match_reference(result, ref_summary):
    pd.testing.assert_frame_equal(result.summary.reset_index(drop=True), ref_summary[sg.SUMMARY_STRUCTURAL_COLUMNS],
                                  check_exact=True)
    assert not set(sg.SUMMARY_LABEL_COLUMNS) & set(result.summary.columns)
    assert not set(sg.SUMMARY_LABEL_COLUMNS) & set(result.assignments.columns)


def test_summary_label_derived_fields_match_reference_separately(evaluation, ref_summary):
    pd.testing.assert_frame_equal(evaluation[1].reset_index(drop=True), ref_summary, check_exact=True)
    assert evaluation[1]["type"].value_counts().to_dict() == {"normal": 10780, "attack-containing": 2000}


def test_summary_is_consistent_with_assignments(result):
    counts = pd.Series(result.scenario).value_counts()
    s = result.summary
    assert (counts.reindex(s["scenario_id"]).to_numpy() == s["n_events"].to_numpy()).all()
    assert (s["n_host"] + s["n_iot"] == s["n_events"]).all()
    assert (s["has_cross_domain_link"] == ((s["n_host"] > 0) & (s["n_iot"] > 0))).all()
    assert s["scenario_id"].is_unique


# --------------------------------------------------------------------------- grouping rules
def test_event_coverage_and_row_identity(result, data):
    n = len(data.host) + len(data.iot)
    assert len(result.assignments) == n == 198360 and result.assignments["node"].is_unique
    expect = np.concatenate(["host:" + data.host["row"].astype(str).to_numpy(),
                             "iot:" + data.iot["row"].astype(str).to_numpy()])
    assert np.array_equal(result.assignments["node"].to_numpy(), expect)       # original 0-based file rows


def test_scenario_counts_and_sizes_have_no_fixed_size_requirement(result):
    size = np.bincount(result.scenario)
    assert len(size) == 109445 and int((size == 1).sum()) == 96665
    assert int((size >= 2).sum()) == 12780 == len(result.summary)
    assert int(size[size >= 2].sum()) == 101695
    assert size[size >= 2].min() == 2 and size.max() == 118 and np.median(size[size >= 2]) == 6
    vc = pd.Series(size[size >= 2]).value_counts()
    assert vc[2] == 108 and vc[3] == 2287 and (vc.index > 6).any()             # sizes 2..118, not 4-6 only


def test_fifteen_minute_continuity_inside_scenarios(result):
    order = np.lexsort((result.nodes["ts_s"].to_numpy(), result.scenario))
    sc, ts = result.scenario[order], result.nodes["ts_s"].to_numpy()[order]
    same = sc[1:] == sc[:-1]
    assert (ts[1:] - ts[:-1])[same].max() <= sg.GATE_GAP
    assert (result.pairs["gap"] <= sg.GATE_GAP).all()


def test_threshold_sweep_for_link_set_c_matches_saved_sweep(result):
    sweep = pd.read_csv(REF / "threshold_sweep_v6v2.csv")
    sweep = sweep[sweep["links"].str.startswith("C")].set_index("T")
    ts_s = result.nodes["ts_s"].to_numpy()
    for T in (2, 3, 4):
        row = sg.sweep_row(sg.group_scenarios(result.pairs, len(result.nodes), ts_s, T), T)
        assert row["scenarios_2plus"] == sweep.loc[T, "scenarios_2plus"]
        assert row["events_in_2plus"] == sweep.loc[T, "events_in_2plus"]
        assert row["max_size"] == sweep.loc[T, "max_size"]
        assert row["scenarios_over_200"] == sweep.loc[T, "over_200"]
    assert result.threshold == 3


# --------------------------------------------------------------------------- ground-truth separation
def test_structural_twins_equal_originals_minus_label(data):
    host, iot = data.to_stage3_frames(include_ground_truth=True)       # evaluation-style frames WITH labels
    managed = set(host["host_ip"])
    orig = sg.node_table(host, iot, managed)
    pd.testing.assert_frame_equal(sg.node_table_structural(*data.to_stage3_frames(), managed),
                                  orig.drop(columns="label"), check_exact=True)


def test_runtime_grouping_code_never_references_ground_truth():
    tree = ast.parse(Path(sg.__file__).read_text(encoding="utf-8").split(MARKER)[1])
    runtime = {"node_table_structural", "scenario_table_structural", "combined_links", "_summary", "run_stage3b"}
    banned = {"label", "type", "attack_type", "predicted_class", "n_label1", "label1_share"}
    seen = set()
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in runtime]:
        seen.add(fn.name)
        doc = ast.get_docstring(fn, clean=False)
        for n in ast.walk(fn):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value != doc:
                if fn.name == "_summary" and n.value in {"type", "label1_share"}:
                    continue          # column pass-through only when the caller supplies a labelled table
                assert n.value not in banned, (fn.name, n.value)
            assert not (isinstance(n, ast.Name) and n.id.startswith("node_labels")), fn.name
    assert seen == runtime


def test_grouping_unchanged_when_ground_truth_is_shuffled(data, result):
    """Run the ORIGINAL label-reading node_table with shuffled labels: connectivity and ids must not change."""
    host, iot = data.to_stage3_frames(include_ground_truth=True)
    rng = np.random.default_rng(123)
    for df, cols in ((host, s2.HOST_TRUTH_COLS), (iot, s2.IOT_TRUTH_COLS)):
        perm = rng.permutation(len(df))
        for c in cols:
            df[c] = df[c].to_numpy()[perm]
    managed = set(host["host_ip"])
    nodes = sg.node_table(host, iot, managed)
    assert not (nodes["label"].to_numpy() == sg.node_labels_for_evaluation(data)).all()    # labels really changed
    links = pd.concat([rm.build_links_all(host, iot, managed), sg.handoff_links(host, rm.service_pids(host)),
                       sg.remote_source_links(host)], ignore_index=True)
    pairs = sg.pair_scores(links, nodes, len(host))
    scenario = sg.group_scenarios(pairs, len(nodes), nodes["ts_s"].to_numpy(), sg.T_FINAL)
    assert np.array_equal(scenario, result.scenario)


def test_grouping_unchanged_when_ground_truth_is_removed(data, result):
    """Unlabelled inputs (no label/type/attack_type columns at all) must give identical assignments."""
    h = data.host.drop(columns=["ts", "row"]).astype({"process_id": str, "parent_process_id": str})
    i = data.iot.drop(columns=["ts", "row"])
    unlabelled = s2.validate_and_build(h, i, allow_missing_ground_truth=True)
    assert list(unlabelled.host_truth.columns) == [] and list(unlabelled.iot_truth.columns) == []
    res2 = sg.run_stage3b(unlabelled)
    pd.testing.assert_frame_equal(res2.assignments, result.assignments, check_exact=True)
    pd.testing.assert_frame_equal(res2.summary, result.summary, check_exact=True)
    pd.testing.assert_frame_equal(res2.links, result.links, check_exact=True)


# --------------------------------------------------------------------------- Stage 4 hand-off contract
def test_stage4_inputs_contract(result):
    s4 = result.stage4_inputs()
    assert set(s4) == {"assign", "summ", "links"}
    assert list(s4["assign"].columns) == ["node", "scenario_id"]
    assert set(s4["summ"]["scenario_id"]) <= set(s4["assign"]["scenario_id"])
    assert (s4["summ"]["n_events"] >= 2).all()
    assert list(s4["links"].columns) == rm.LINK_COLUMNS


def test_stage4_edges_possible_for_every_scenario(result):
    """Stage 4 asserts every scenario has >= 1 intra-scenario link; verify this holds for the grouping."""
    n_host = int((result.nodes["kind"] == "host").sum())
    a = sg.node_index(result.links["node_a"], n_host)
    b = sg.node_index(result.links["node_b"], n_host)
    same = result.scenario[a] == result.scenario[b]
    with_edge = set(result.scenario[a][same])
    assert set(result.summary["scenario_id"]) == with_edge


def test_scenario_membership_matches_saved_stage4_graph_nodes(result):
    """Saved Stage 4 node table (node -> scenario_id) must equal our assignments for the >= 2-event scenarios."""
    nodes = pd.read_csv(STAGES / "Stage 4" / "Outputs" / "Graph Data" / "final_node_labels_v6v2 (1).csv",
                        usecols=["node", "scenario_id"])
    ours = result.assignments[result.assignments["scenario_id"].isin(result.summary["scenario_id"])]
    merged = ours.merge(nodes, on="node", suffixes=("_ours", "_s4"))
    assert len(nodes) == len(ours) == len(merged) == 101695
    assert (merged["scenario_id_ours"] == merged["scenario_id_s4"]).all()
