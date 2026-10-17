"""Tests for xdr_c3.relationship_mapping on the real Host / IoT data, against the saved Stage 3 reference."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xdr_c3 import relationship_mapping as rm
from xdr_c3 import stage2_validate as s2
from xdr_c3.config import PROJECT_ROOT

STAGES = PROJECT_ROOT / "Stages"
NB3 = STAGES / "Stage 3" / "Stage3_Cross_Domain_Relationship_Mapping_V6v2.ipynb"
REF3 = STAGES / "Stage 3" / "Outputs" / "links_all_v6v2.csv"
REF3B = STAGES / "Stage 3b" / "Outputs" / "links_stage3b_v6v2.csv"
MARKER = "# =========================================================================== application integration"


@pytest.fixture(scope="module")
def data():
    return s2.load_and_validate()


@pytest.fixture(scope="module")
def links(data):
    return rm.build_links_from_stage2(data)


@pytest.fixture(scope="module")
def ref():
    return pd.read_csv(REF3)


# --------------------------------------------------------------------------- extraction is verbatim
def test_module_is_verbatim_notebook_code():
    nb = json.loads(NB3.read_text(encoding="utf-8"))
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "".join(c["source"]).startswith("%%writefile relationship_mapping.py"))
    original = cell.split("\n", 1)[1].rstrip("\n")
    ours = Path(rm.__file__).read_text(encoding="utf-8")
    assert ours.split(MARKER)[0].rstrip("\n") == original


def test_original_constants_unchanged():
    assert (rm.W_MATCH, rm.HOST_HOST_TOL, rm.FLOW_GAP, rm.ENTRY_GAP, rm.FALLBACK_GAP, rm.SERVICE_MIN_EVENTS) == \
           (30, 5, 900, 900, 5, 30)
    assert rm.FMT == "%d/%m/%Y %H:%M:%S" == s2.FMT
    assert rm.KEYS == ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]
    assert rm.NET_EVENTS == ["network_connect", "network_accept"]


def test_module_never_reads_ground_truth_columns():
    import ast
    tree = ast.parse(Path(rm.__file__).read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node, clean=False):
            docstrings.add(id(node.body[0].value))
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings}
    assert not literals & {"label", "type", "attack_type", "predicted_class", "threat_state", "trace_mode"}


# --------------------------------------------------------------------------- equivalence with the saved reference
def test_output_structure(links, ref):
    assert list(links.columns) == rm.LINK_COLUMNS == list(ref.columns)
    assert len(links) == len(ref) == 91463


def test_link_type_distribution_matches_reference(links, ref):
    got = links.link_type.value_counts().sort_index().to_dict()
    assert got == ref.link_type.value_counts().sort_index().to_dict()
    assert got == {"connect_accept": 8171, "entry_to_child": 11877, "fallback_accept": 695, "flow_same_dst": 1023,
                   "flow_same_src": 5892, "flow_same_src_dst": 7411, "iot_host_match": 5620, "parent_child": 3286,
                   "same_process": 47488}


def test_full_row_level_equivalence_in_original_order(links, ref, tmp_path):
    out = tmp_path / "links.csv"                      # written outside the project
    links.to_csv(out, index=False, lineterminator="\n")
    back = pd.read_csv(out)
    pd.testing.assert_frame_equal(back, ref, check_exact=True)
    assert out.read_bytes() == REF3.read_bytes()       # byte-identical file


def test_node_pairs_points_and_gaps_identical(links, ref):
    key = ["node_a", "node_b", "link_type"]
    assert not links.duplicated(key).any()
    m = links.merge(ref, on=key, suffixes=("_g", "_r"))
    assert len(m) == len(ref)
    assert (m.points_g == m.points_r).all()
    assert (m.gap_sec_g == m.gap_sec_r).all()          # exact, not approximate


# --------------------------------------------------------------------------- notebook sanity rules (Stage 3, Step 10)
def test_notebook_sanity_checks(links, data):
    n_host, n_iot = len(data.host), len(data.iot)
    for col in ("node_a", "node_b"):
        kind = links[col].str.split(":").str[0]
        row = links[col].str.split(":").str[1].astype(int)
        assert (((kind == "host") & (row < n_host)) | ((kind == "iot") & (row < n_iot))).all()
    assert (links.node_a != links.node_b).all()
    t = links.link_type
    assert (links[t == "connect_accept"].gap_sec <= 5).all()
    assert (links[t == "iot_host_match"].gap_sec <= 30).all()
    assert (links[~t.isin(["connect_accept", "iot_host_match", "fallback_accept"])].gap_sec <= 900).all()
    pts = links.groupby("link_type").points.agg(lambda s: sorted(s.unique())).to_dict()
    assert pts == {"connect_accept": [3], "entry_to_child": [2, 3], "fallback_accept": [2], "flow_same_dst": [1],
                   "flow_same_src": [1], "flow_same_src_dst": [2], "iot_host_match": [3], "parent_child": [2],
                   "same_process": [2]}


# --------------------------------------------------------------------------- individual stages on real data
def test_candidate_pools_and_one_to_one_matching(data):
    host, iot = data.to_stage3_frames()
    managed = set(host["host_ip"])
    net = rm.network_events(host)
    assert net["event_type"].isin(rm.NET_EVENTS).all()
    assert ((net["src_ip"] == net["host_ip"]) | (net["dst_ip"] == net["host_ip"])).all()
    rel = rm.relevant_flows(iot, managed)
    assert (rel["src_ip"].isin(managed) | rel["dst_ip"].isin(managed)).all()
    cand = rm.candidate_pairs(rel, net)
    assert set(cand["orient"]) == {"direct", "reversed"}
    m = rm.match_iot_host(host, iot, managed)
    assert m["iot_row"].is_unique and m["host_row"].is_unique
    assert (m["dt_sec"] <= rm.W_MATCH).all() and len(m) == 5620
    assert m["iot_ts"].is_monotonic_increasing


def test_host_host_and_flow_and_process_links(data):
    host, iot = data.to_stage3_frames()
    managed = set(host["host_ip"])
    hh = rm.host_host_links(host, managed)
    assert len(hh) == 8171 and (hh.points == 3).all() and (hh.gap_sec <= rm.HOST_HOST_TOL).all()
    fl = rm.flow_links(iot)
    assert set(fl.link_type) == {"flow_same_src_dst", "flow_same_src", "flow_same_dst"}
    services = rm.service_pids(host)
    counts = host["process_id"].value_counts()
    assert 1 in services and services == set(counts[counts >= rm.SERVICE_MIN_EVENTS].index) | {1}
    pl = rm.process_links(host, services)
    assert set(pl.link_type) == {"same_process", "parent_child", "entry_to_child", "fallback_accept"}


# --------------------------------------------------------------------------- integration with Stage 2
def test_uses_stage2_rows_and_timestamps_without_reparsing(data):
    host, iot = data.to_stage3_frames()
    assert host["ts"].equals(data.host["ts"]) and np.array_equal(host["row"].to_numpy(), np.arange(len(host)))
    before_h, before_i = data.host.copy(), data.iot.copy()
    rm.build_links_from_stage2(data)
    pd.testing.assert_frame_equal(data.host, before_h)  # inputs are not mutated
    pd.testing.assert_frame_equal(data.iot, before_i)


def test_links_do_not_depend_on_ground_truth(data, links):
    """Same output whether ground truth is absent, present, or shuffled (the algorithm never reads it)."""
    host, iot = data.to_stage3_frames(include_ground_truth=True)
    rng = np.random.default_rng(0)
    for df, cols in ((host, s2.HOST_TRUTH_COLS), (iot, s2.IOT_TRUTH_COLS)):
        perm = rng.permutation(len(df))
        for c in cols:
            df[c] = df[c].to_numpy()[perm]
    shuffled = rm.build_links_all(host, iot, set(host["host_ip"]))
    pd.testing.assert_frame_equal(shuffled, links, check_exact=True)


def test_adapter_rejects_frames_containing_ground_truth(data):
    host, iot = data.to_stage3_frames(include_ground_truth=True)
    leaky = s2.Stage2Data(host, iot, data.host_truth, data.iot_truth, data.report)
    with pytest.raises(ValueError):
        rm.build_links_from_stage2(leaky)


# --------------------------------------------------------------------------- Stage 3b input contract
def test_output_satisfies_stage3b_contract(links, data):
    """Stage 3b consumes links with node_a / node_b / link_type / points / gap_sec, node ids 'host:<row>' /
    'iot:<row>' and integer points. Its saved link set is ours plus the two Stage 3b rules."""
    assert links["points"].dtype.kind == "i" and links["gap_sec"].dtype == np.float64
    assert links["node_a"].str.fullmatch(r"(host|iot):\d+").all() and links["node_b"].str.fullmatch(r"(host|iot):\d+").all()
    ref3b = pd.read_csv(REF3B)
    key = ["node_a", "node_b", "link_type"]
    ours = set(map(tuple, links[key].to_numpy()))
    theirs = set(map(tuple, ref3b[key].to_numpy()))
    assert ours <= theirs
    extra = ref3b[~ref3b.set_index(key).index.isin(links.set_index(key).index)]
    assert set(extra.link_type) == {"service_handoff", "same_remote_source"}
    assert len(ref3b) == len(links) + len(extra) == 99095
    # Stage 3b node_index() semantics: ids map to positions in [host rows | iot rows]
    n_host = len(data.host)
    kind = links["node_a"].str.split(":").str[0].to_numpy()
    row = links["node_a"].str.split(":").str[1].astype(int).to_numpy()
    idx = np.where(kind == "host", row, n_host + row)
    assert idx.min() >= 0 and idx.max() < n_host + len(data.iot)
