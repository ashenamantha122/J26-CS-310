"""Tests for xdr_c3.stage2_validate: real-data validation, Stage 3 compatibility, and negative cases."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from xdr_c3 import stage2_validate as s2
from xdr_c3.config import HOST_CSV, IOT_CSV


# --------------------------------------------------------------------------- reference: Stage 3 load_data, copied verbatim
def stage3_load_data(host_csv, iot_csv):
    """relationship_mapping.load_data from the Stage 3 notebook (path handling simplified)."""
    FMT = "%d/%m/%Y %H:%M:%S"
    host = pd.read_csv(host_csv, dtype=str, keep_default_na=False)
    iot = pd.read_csv(iot_csv, dtype=str, keep_default_na=False)
    host["ts"] = pd.to_datetime(host["timestamp"], format=FMT)
    iot["ts"] = pd.to_datetime(iot["timestamp"], format=FMT)
    host["row"] = np.arange(len(host))
    iot["row"] = np.arange(len(iot))
    for c in ["process_id", "parent_process_id"]:
        host[c] = host[c].astype("int64")
    return host, iot


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def real():
    h_before, i_before = sha256(HOST_CSV), sha256(IOT_CSV)
    data = s2.load_and_validate()
    return data, (h_before, i_before)


@pytest.fixture()
def raw():
    """A small, real slice of both files (raw text frames) to mutate in negative tests."""
    host = pd.read_csv(HOST_CSV, dtype=str, keep_default_na=False, nrows=3000)
    iot = pd.read_csv(IOT_CSV, dtype=str, keep_default_na=False, nrows=1000)
    return host, iot


def failed_names(report):
    return {c.name for c in report.errors}


# --------------------------------------------------------------------------- real data
def test_real_data_passes(real):
    data, _ = real
    assert data.report.ok, data.report.summary()
    assert data.report.stats == {"host_rows": 138945, "host_columns": 26, "iot_rows": 59415, "iot_columns": 22}
    assert len(data.host) == 138945 and len(data.iot) == 59415


def test_real_data_files_not_modified(real):
    _, (h_before, i_before) = real
    assert sha256(HOST_CSV) == h_before and sha256(IOT_CSV) == i_before


def test_real_data_matches_stage3_load_data_exactly(real):
    data, _ = real
    ref_host, ref_iot = stage3_load_data(HOST_CSV, IOT_CSV)
    host, iot = data.to_stage3_frames(include_ground_truth=True)
    pd.testing.assert_frame_equal(host, ref_host, check_exact=True)
    pd.testing.assert_frame_equal(iot, ref_iot, check_exact=True)
    assert list(host.columns) == s2.HOST_COLS + ["ts", "row"]
    assert list(iot.columns) == s2.IOT_COLS + ["ts", "row"]


def test_real_row_ids_are_stable_file_positions(real):
    data, _ = real
    assert np.array_equal(data.host["row"].to_numpy(), np.arange(len(data.host)))
    assert np.array_equal(data.iot["row"].to_numpy(), np.arange(len(data.iot)))
    assert np.array_equal(data.host_truth.index.to_numpy(), data.host["row"].to_numpy())
    assert np.array_equal(data.iot_truth.index.to_numpy(), data.iot["row"].to_numpy())
    assert data.host["process_id"].dtype == np.int64 and data.host["parent_process_id"].dtype == np.int64


def test_real_ground_truth_is_separate(real):
    data, _ = real
    for df in data.to_stage3_frames():
        s2.assert_no_ground_truth(df)
    assert list(data.host_truth.columns) == ["label", "type"]
    assert list(data.iot_truth.columns) == ["label", "attack_type"]
    for leaked in ("label", "type", "attack_type"):
        assert leaked not in data.host.columns and leaked not in data.iot.columns
    with pytest.raises(ValueError):
        s2.assert_no_ground_truth(data.to_stage3_frames(include_ground_truth=True)[0])


def test_real_placeholders_preserved(real):
    data, _ = real
    h = data.host
    mem = h[h["event_type"] == "memory_event"]
    assert (mem["src_ip"] == "-").all() and (mem["src_port"] == "").all()   # '-' and blank kept as-is
    assert not h.isna().any().any() and not data.iot.isna().any().any()    # nothing imputed / NaN
    assert (h.loc[h["event_type"] == "file_event", "file_operation"] != "-").all()


def test_real_known_counts_from_notebook(real):
    data, _ = real
    assert int(data.host_truth["label"].astype(int).sum()) == 18149
    assert int(data.iot_truth["label"].astype(int).sum()) == 9841


def test_unlabelled_inference_input_accepted(real):
    data, _ = real
    h = data.host.drop(columns=["ts", "row"]).head(500)
    i = data.iot.drop(columns=["ts", "row"]).head(500)
    # restore text form of the process ids (they are int64 in the validated frames)
    h = h.astype({"process_id": str, "parent_process_id": str})
    out = s2.validate_and_build(h, i, allow_missing_ground_truth=True)
    assert out.report.ok and list(out.host_truth.columns) == []
    # the same frames are rejected under the strict (labelled) schema
    with pytest.raises(s2.Stage2ValidationError):
        s2.validate_and_build(h, i)


# --------------------------------------------------------------------------- slice sanity and non-mutation
def test_slice_passes_and_inputs_not_mutated(raw):
    host, iot = raw
    h0, i0 = host.copy(), iot.copy()
    out = s2.validate_and_build(host, iot)
    assert out.report.ok, out.report.summary()
    pd.testing.assert_frame_equal(host, h0)
    pd.testing.assert_frame_equal(iot, i0)


# --------------------------------------------------------------------------- negative: schema / parsing
def test_rejects_missing_column(raw):
    host, iot = raw
    rep = s2.validate_frames(host.drop(columns=["cpu_usage"]), iot)
    assert not rep.ok and "host columns exactly as expected" in failed_names(rep)


def test_rejects_wrong_column_order(raw):
    host, iot = raw
    cols = list(host.columns)
    cols[0], cols[1] = cols[1], cols[0]
    assert "host columns exactly as expected" in failed_names(s2.validate_frames(host[cols], iot))


def test_rejects_extra_column(raw):
    host, iot = raw
    iot = iot.assign(scenario_id="1")
    assert "iot columns exactly as expected" in failed_names(s2.validate_frames(host, iot))


@pytest.mark.parametrize("bad", ["2019-01-04 00:06:37", "31/02/2019 00:06:37", "04/01/2019 00:06", "garbage", ""])
def test_rejects_bad_timestamp(raw, bad):
    host, iot = raw
    host = host.copy()
    host.loc[5, "timestamp"] = bad
    rep = s2.validate_frames(host, iot)
    assert any(c.name.startswith("host timestamps parse") for c in rep.errors)


def test_unsorted_timestamps_only_warn(raw):
    host, iot = raw
    host = host.iloc[::-1].reset_index(drop=True)
    rep = s2.validate_frames(host, iot)
    assert rep.ok and any("time order" in c.name for c in rep.warnings)


def test_rejects_non_integer_process_id(raw):
    host, iot = raw
    host = host.copy()
    host.loc[3, "process_id"] = "abc"
    assert "host process_id is integer text" in failed_names(s2.validate_frames(host, iot))


# --------------------------------------------------------------------------- negative: quality rules
def test_rejects_duplicate_rows(raw):
    host, iot = raw
    host = pd.concat([host, host.iloc[[0]]], ignore_index=True)
    assert "no duplicate rows (host / iot)" in failed_names(s2.validate_frames(host, iot))


def test_rejects_invalid_ip(raw):
    host, iot = raw
    iot = iot.copy()
    iot.loc[0, "src_ip"] = "999.1.1.1"
    assert "no invalid IP strings (host / iot)" in failed_names(s2.validate_frames(host, iot))


def test_accepts_placeholder_ip_values(raw):
    host, iot = raw
    assert (host["src_ip"] == "-").any()           # '-' is valid "not applicable", not an invalid IP
    assert s2.validate_frames(host, iot).ok


def test_rejects_unparseable_and_negative_numbers(raw):
    host, iot = raw
    h = host.copy()
    h.loc[2, "cpu_usage"] = "high"
    assert "numeric columns all parse (host / iot)" in failed_names(s2.validate_frames(h, iot))
    i = iot.copy()
    i.loc[2, "src_bytes"] = "-5"
    assert "no negative numbers (host / iot)" in failed_names(s2.validate_frames(host, i))


def test_rejects_port_out_of_range(raw):
    host, iot = raw
    i = iot.copy()
    i.loc[1, "dst_port"] = "70000"
    assert "ports inside 0-65535 (host / iot)" in failed_names(s2.validate_frames(host, i))


def test_rejects_score_out_of_range(raw):
    host, iot = raw
    h = host.copy()
    h.loc[4, "threat_confidence_score"] = "1.5"
    assert "scores inside [0,1] (host / iot)" in failed_names(s2.validate_frames(h, iot))


def test_rejects_incomplete_network_tuple_and_stray_tuple(raw):
    host, iot = raw
    h = host.copy()
    net_idx = h.index[h["event_type"] == "network_accept"][0]
    h.loc[net_idx, "dst_port"] = ""
    assert "network events carry a complete tuple" in failed_names(s2.validate_frames(h, iot))
    h = host.copy()
    mem_idx = h.index[h["event_type"] == "memory_event"][0]
    h.loc[mem_idx, "src_ip"] = "10.20.1.11"
    assert "non-network events carry no tuple" in failed_names(s2.validate_frames(h, iot))


# --------------------------------------------------------------------------- negative: ground-truth rules
def test_rejects_bad_label_and_inconsistent_type(raw):
    host, iot = raw
    h = host.copy()
    h.loc[0, "label"] = "2"
    assert "label values only 0/1 (host / iot)" in failed_names(s2.validate_frames(h, iot))
    h = host.copy()
    h.loc[0, "label"] = "1"                     # normal row relabelled attack but type stays 'normal'
    assert "host type == 'normal' exactly when label == 0" in failed_names(s2.validate_frames(h, iot))


def test_rejects_banned_category(raw):
    host, iot = raw
    h = host.copy()
    h.loc[0, "type"] = "lateral_movement"
    h.loc[0, "label"] = "1"
    assert "no banned row-level categories" in failed_names(s2.validate_frames(h, iot))


# --------------------------------------------------------------------------- error handling
def test_validate_and_build_raises_with_report(raw):
    host, iot = raw
    h = host.copy()
    h.loc[0, "timestamp"] = "bad"
    with pytest.raises(s2.Stage2ValidationError) as exc:
        s2.validate_and_build(h, iot)
    assert not exc.value.report.ok
    out = s2.validate_and_build(h, iot, raise_on_error=False)
    assert out.host is None and not out.report.ok


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        s2.load_and_validate(tmp_path / "nope.csv", IOT_CSV)
