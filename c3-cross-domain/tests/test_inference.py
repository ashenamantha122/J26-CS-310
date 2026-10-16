"""Tests for the saved R-GCN-GAE inference (xdr_c3.inference, stage5_model, stage5_utils) on the real artifacts."""
import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from xdr_c3 import config
from xdr_c3 import inference as inf
from xdr_c3 import stage5_model as m
from xdr_c3 import stage5_utils as u
from xdr_c3.config import SPLIT_NPZ, STAGE5_DIR, VALIDATION_SCORES_CSV
from xdr_c3.model_contract import load_contract

TOL = 1e-5          # absolute score tolerance: observed CPU-vs-Colab float32 noise is ~3e-7 (see verification/stage5)
THRESHOLD = 2.5825197985552437


@pytest.fixture(scope="session")
def val_ref():
    return pd.read_csv(VALIDATION_SCORES_CSV).set_index("scenario_id")


@pytest.fixture(scope="session")
def val_result(detector):
    return detector.predict_arrays(inf.batch_from_npz(SPLIT_NPZ["validation"]))


@pytest.fixture()
def small(graphs):
    """A few real graphs as Stage 4 arrays (label-free)."""
    return graphs.arrays(graphs.scenario_ids[[3, 400, 2000, 9000, 12000]])


# =========================================================================== source and checkpoint compatibility
def test_application_modules_are_the_original_stage5_code():
    for ours, original in (("stage5_utils.py", "stage5_utils (1).py"), ("stage5_model.py", "stage5_model (1).py")):
        a = (Path(m.__file__).parent / ours).read_text(encoding="utf-8")
        b = (STAGE5_DIR / original).read_text(encoding="utf-8").replace("\r\n", "\n")
        if ours == "stage5_model.py":
            b = b.replace("import stage5_utils as u\n", "from . import stage5_utils as u   # packaging change only (was: import stage5_utils as u)\n")
        assert a == b


def test_source_is_rgcn_not_graphsage(detector):
    assert hasattr(m, "RGCNGAE") and hasattr(m, "RGCNLayer")
    assert not hasattr(m, "SAGEGAE") and "SAGE" not in Path(m.__file__).read_text() and "GINE" not in Path(m.__file__).read_text()
    assert isinstance(detector.model, m.RGCNGAE)


def test_checkpoint_loaded_trusted_and_in_eval_mode(detector):
    assert detector.checkpoint_sha256 == config.CHECKPOINT_SHA256 == inf.sha256_file(config.CHECKPOINT_PATH)
    assert detector.model.training is False                      # evaluation mode (dropout disabled)
    assert sum(p.numel() for p in detector.model.parameters()) == 96515
    cfg = detector.cfg
    assert (cfg["kind"], cfg["epoch"], cfg["f_in"], cfg["e_dim"], cfg["hidden"], cfg["latent"], cfg["dropout"],
            cfg["setting"]) == ("rgcn", 40, 84, 14, 64, 32, 0.1, "nocomp")
    assert detector.model.n_relations == 10 and detector.model.rel_cols.tolist() == list(range(4, 14))


def test_checkpoint_hash_mismatch_is_refused(tmp_path):
    bad = tmp_path / "epoch_040.pt"
    shutil.copy(config.CHECKPOINT_PATH, bad)
    with bad.open("ab") as fh:
        fh.write(b"x")
    with pytest.raises(inf.ArtifactMismatchError):
        inf.load_trusted_checkpoint(bad)


def test_checkpoint_loaded_without_unrestricted_pickle(monkeypatch):
    seen = {}
    real = torch.load

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(inf.torch, "load", spy)
    inf.load_trusted_checkpoint()
    assert seen.get("weights_only") is True


def test_checkpoint_matches_saved_feature_and_relation_contract(detector):
    c = load_contract()
    assert tuple(detector.cols) == c.model_cols and len(c.model_cols) == 84 and len(c.feature_names) == 92
    assert tuple(detector.cfg["rel_names"]) == c.relation_names == tuple(
        json.loads(config.MODEL_CONFIG_JSON.read_text())["relations"])
    assert len(c.edge_feature_names) == 14
    npz = np.load(SPLIT_NPZ["train"])
    assert [str(s) for s in npz["feature_names"]] == list(c.feature_names)
    assert [str(s) for s in npz["edge_feature_names"]] == list(c.edge_feature_names)
    # the scaler/relation contract of the graphs fed to the model is the one validated in Stage 4
    assert c.node_scaler and tuple(c.edge_scaler) == ("e_score", "e_log_gap")


# =========================================================================== locked configuration
def test_locked_selection_values(detector):
    lock = inf.load_locked_selection()
    assert (lock["selected_epoch"], lock["score"], lock["knn_k"], lock["threshold"]) == (40, "knn_embedding", 5, THRESHOLD)
    assert detector.threshold == THRESHOLD and detector.k == 5 and detector.knn.k == 5


def test_tampered_lock_is_rejected(tmp_path):
    lock = json.loads(config.LOCKED_SELECTION_JSON.read_text())
    lock["threshold"] = 2.0
    p = tmp_path / "lock.json"
    p.write_text(json.dumps(lock))
    with pytest.raises(inf.ArtifactMismatchError):
        inf.load_locked_selection(p)


def test_checkpoint_contract_mismatch_is_rejected(detector):
    cfg = dict(detector.cfg, f_in=91)
    with pytest.raises(inf.ArtifactMismatchError):
        inf.check_checkpoint_against_contract(cfg, detector.contract, detector.lock)
    cfg = dict(detector.cfg, rel_names=list(reversed(detector.cfg["rel_names"])))
    with pytest.raises(inf.ArtifactMismatchError):
        inf.check_checkpoint_against_contract(cfg, detector.contract, detector.lock)


# =========================================================================== normal-reference embeddings and kNN
def test_reference_embeddings_shape_and_normality(detector):
    e = detector.reference_embeddings
    assert e.shape == (8946, 32) and np.isfinite(e).all()
    with np.load(SPLIT_NPZ["train"]) as z:                       # reference-integrity check only (not an inference input)
        assert int(z["scenario_is_attack"].sum()) == 0 and len(z["scenario_id"]) == 8946


def test_reference_embeddings_are_reproducible(detector):
    again = inf.Stage5Detector.load(use_cache=False)
    assert np.array_equal(again.reference_embeddings, detector.reference_embeddings)


def test_knn_score_is_mean_distance_to_5_nearest_standardised_normals(detector, val_result):
    ref = detector.reference_embeddings
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd < 1e-6] = 1.0
    z = (ref - mu) / sd
    for i in (0, 17, 400, 1000):
        q = (val_result.embeddings[i] - mu) / sd
        brute = np.sort(np.linalg.norm(z - q, axis=1))[:5].mean()
        assert abs(brute - val_result.scores["anomaly_score"].iloc[i]) < 1e-9


def test_reference_cache_roundtrip_and_stale_detection(tmp_path, detector, monkeypatch):
    first = inf.Stage5Detector.load(use_cache=True, cache_dir=tmp_path)
    assert np.array_equal(first.reference_embeddings, detector.reference_embeddings)
    calls = []
    real = m.score_bucket
    monkeypatch.setattr(inf.m, "score_bucket", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    second = inf.Stage5Detector.load(use_cache=True, cache_dir=tmp_path)           # served from cache
    assert calls == [] and np.array_equal(second.reference_embeddings, first.reference_embeddings)
    meta_p = tmp_path / "reference_embeddings.json"
    meta = json.loads(meta_p.read_text())
    assert {"checkpoint_sha256", "train_npz_sha256", "feature_names_sha256", "torch", "embeddings_sha256"} <= set(meta)
    meta["checkpoint_sha256"] = "0" * 64                                            # stale: different checkpoint
    meta_p.write_text(json.dumps(meta))
    inf.Stage5Detector.load(use_cache=True, cache_dir=tmp_path)
    assert calls == [1]                                                             # recomputed
    with np.load(tmp_path / "reference_embeddings.npz") as z:                       # rewritten cache is valid again
        assert z["embeddings"].shape == (8946, 32)
    emb = np.load(tmp_path / "reference_embeddings.npz")["embeddings"].copy()
    emb[0, 0] += 1.0
    np.savez(tmp_path / "reference_embeddings.npz", embeddings=emb)                 # corrupted content
    again = inf.Stage5Detector.load(use_cache=True, cache_dir=tmp_path)
    assert calls == [1, 1] and np.array_equal(again.reference_embeddings, detector.reference_embeddings)


def test_original_research_artifacts_untouched_by_cache(detector):
    assert config.CACHE_DIR.parent == config.PROJECT_ROOT / "cache"
    assert config.STAGES_DIR not in config.CACHE_DIR.parents


# =========================================================================== decisions
def test_decision_rule_is_strictly_greater_than_threshold(val_result):
    s = val_result.scores
    assert (s["prediction"] == np.where(s["anomaly_score"] > THRESHOLD, "Suspicious", "Normal")).all()
    assert (s["is_suspicious"] == (s["anomaly_score"] > THRESHOLD)).all() and (s["threshold"] == THRESHOLD).all()
    assert set(s["prediction"]) == {"Normal", "Suspicious"}


def test_score_equal_to_threshold_is_normal(detector, val_result):
    sid = int(val_result.scores["scenario_id"].iloc[5])
    score = float(val_result.scores["anomaly_score"].iloc[5])
    old = detector.threshold
    try:
        detector.threshold = score                                   # strict '>' keeps an equal score Normal
        r = detector.predict_arrays(inf.batch_from_npz(SPLIT_NPZ["validation"]))
        row = r.scores.set_index("scenario_id").loc[sid]
        assert row["prediction"] == "Normal"
    finally:
        detector.threshold = old


# =========================================================================== label-free inputs and leakage
def test_label_free_batch_needs_no_label_arrays(small):
    assert not {"node_label", "scenario_is_attack"} & set(small)
    b = inf.batch_from_arrays(small)
    assert not hasattr(b, "node_label") and not hasattr(b, "is_attack")
    assert np.array_equal(b.x, small["x"]) and np.array_equal(b.edge_attr, small["edge_attr"])      # values untouched
    assert np.array_equal(b.scenario_id, small["scenario_id"]) and np.array_equal(b.node_ptr, small["node_ptr"])


def test_original_bucket_loader_requires_labels_but_ours_does_not(tmp_path, small):
    p = tmp_path / "unlabelled.npz"
    np.savez(p, **small)
    with pytest.raises(KeyError):
        u.Bucket(p)                                                  # original evaluation pathway is unchanged
    assert inf.batch_from_npz(p).n == 5


def test_npz_loader_never_reads_label_arrays(monkeypatch):
    opened = []
    real = np.load

    class Spy:
        def __init__(self, z):
            self.z = z
            self.files = z.files

        def __getitem__(self, k):
            opened.append(k)
            return self.z[k]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.z.close()

    monkeypatch.setattr(inf.np, "load", lambda *a, **k: Spy(real(*a, **k)))
    inf.batch_from_npz(SPLIT_NPZ["validation"])
    assert not {"node_label", "scenario_is_attack"} & set(opened) and "x" in opened


def test_labels_cannot_influence_scores(detector, graphs, small):
    base = detector.predict_arrays(small).scores
    rng = np.random.default_rng(0)
    with_labels = dict(small, node_label=rng.integers(0, 2, len(small["x"])).astype(np.int8),
                       scenario_is_attack=rng.integers(0, 2, len(small["scenario_id"])).astype(np.int8))
    pd.testing.assert_frame_equal(detector.predict_arrays(with_labels).scores.drop(columns=[]), base, check_exact=True)


def test_inference_code_has_no_ground_truth_references():
    tree = ast.parse(Path(inf.__file__).read_text(encoding="utf-8"))
    banned = {"label", "type", "attack_type", "node_label", "is_attack"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in banned:
            raise AssertionError(node.value)
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("predict"):
            assert not {a.arg for a in fn.args.args} & {"labels", "node_label", "scenario_is_attack", "truth"}


# =========================================================================== unsupported input rejection
def _mut(small, **repl):
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in small.items()}
    out.update(repl)
    return out


@pytest.mark.parametrize("name,change", [
    ("missing key", lambda s: {k: v for k, v in s.items() if k != "edge_ptr"}),
    ("wrong node width", lambda s: _mut(s, x=s["x"][:, :91])),
    ("wrong edge width", lambda s: _mut(s, edge_attr=s["edge_attr"][:, :13])),
    ("renamed feature", lambda s: _mut(s, feature_names=np.array(["zz"] + [str(n) for n in s["feature_names"][1:]]))),
    ("swapped feature order", lambda s: _mut(s, feature_names=np.array([str(n) for n in s["feature_names"]][::-1]))),
    ("renamed relation", lambda s: _mut(s, edge_feature_names=np.array(
        [str(n).replace("same_process", "other") for n in s["edge_feature_names"]]))),
    ("pointer not starting at zero", lambda s: _mut(s, node_ptr=s["node_ptr"] + 1)),
    ("pointer total mismatch", lambda s: _mut(s, edge_ptr=np.r_[s["edge_ptr"][:-1], s["edge_ptr"][-1] - 2])),
    ("empty graph", lambda s: _mut(s, node_ptr=np.r_[0, 0, s["node_ptr"][2:]])),
    ("duplicate scenario ids", lambda s: _mut(s, scenario_id=np.full_like(s["scenario_id"], 7))),
    ("edge endpoint out of range", lambda s: _mut(s, edge_index=np.where(np.arange(s["edge_index"].shape[1]) == 0, 10 ** 6, s["edge_index"]))),
    ("NaN feature", lambda s: _mut(s, x=np.where(np.arange(len(s["x"]))[:, None] == 0, np.nan, s["x"]).astype(np.float32))),
])
def test_unsupported_inputs_are_rejected(detector, small, name, change):
    with pytest.raises(inf.InferenceInputError):
        detector.predict_arrays(change(small))


def test_non_one_hot_relation_encoding_is_rejected(detector, small):
    ea = small["edge_attr"].copy()
    ea[0, 4:14] = 0.0                                          # no relation at all
    with pytest.raises(inf.InferenceInputError, match="one-hot"):
        detector.predict_arrays(_mut(small, edge_attr=ea))
    ea = small["edge_attr"].copy()
    ea[0, 4:14] = 0.0
    ea[0, 4], ea[0, 5] = 1.0, 1.0                              # two relations at once
    with pytest.raises(inf.InferenceInputError, match="one-hot"):
        detector.predict_arrays(_mut(small, edge_attr=ea))


def test_odd_directed_edge_count_is_rejected(detector, small):
    ptr = small["edge_ptr"].copy()
    ptr[1:] -= 1
    ptr[-1] = small["edge_ptr"][-1] - 1
    bad = _mut(small, edge_ptr=ptr, edge_index=small["edge_index"][:, :-1], edge_attr=small["edge_attr"][:-1])
    with pytest.raises(inf.InferenceInputError):
        detector.predict_arrays(bad)


def test_model_feature_indices_must_fit_the_matrix(detector, small):
    b = inf.batch_from_arrays(small)
    with pytest.raises(inf.InferenceInputError):
        inf.validate_graph_arrays(b, detector.contract, np.array([0, 1, 500]))


# =========================================================================== reproducibility
def test_scores_are_reproducible_and_batch_size_independent(detector, small):
    a = detector.predict_arrays(small).scores
    b = detector.predict_arrays(small).scores
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    c = detector.predict_arrays(small, batch_size=2).scores
    assert np.abs(a["anomaly_score"] - c["anomaly_score"]).max() < 1e-6
    assert (a["prediction"] == c["prediction"]).all()


# =========================================================================== saved validation / test comparison
def test_validation_scores_match_saved_per_graph_scores(val_result, val_ref):
    s = val_result.scores.set_index("scenario_id")
    assert set(s.index) == set(val_ref.index) and len(s) == len(val_ref) == 1278          # same coverage, by scenario id
    s = s.loc[val_ref.index]
    assert s["threshold"].iloc[0] == THRESHOLD
    diff = (s["anomaly_score"] - val_ref["knn_embedding"]).abs()
    assert diff.max() < TOL, diff.max()
    assert diff.max() < 1e-6                                                              # observed ~2.7e-7
    assert (s["is_suspicious"].to_numpy() == val_ref["predicted_attack"].astype(bool).to_numpy()).all()
    ge = val_result.graph_errors.set_axis(val_result.scores["scenario_id"]).loc[val_ref.index]
    for c in ("recon_total", "node_mean", "node_max", "edge_mean", "edge_max"):
        assert (ge[c] - val_ref[c]).abs().max() < 1e-5, c


def test_validation_boundary_graph_is_documented(val_result, val_ref):
    """The locked threshold equals the score of one normal validation graph, so that graph lies ON the threshold.
    Its decision (Normal, since the rule is '>') is the same as saved; no other graph is within 1e-5."""
    near = (val_ref["knn_embedding"] - THRESHOLD).abs()
    assert int((near < 1e-5).sum()) == 1 and near.idxmin() == 37887
    row = val_result.scores.set_index("scenario_id").loc[37887]
    assert row["prediction"] == "Normal" and abs(row["anomaly_score"] - THRESHOLD) < TOL
    assert near.drop(37887).min() > 1e-4


def test_validation_aggregate_metrics_match_saved(val_result, val_ref):
    with np.load(SPLIT_NPZ["validation"]) as z:                  # labels used ONLY here, for evaluation
        y = pd.Series(z["scenario_is_attack"].astype(int), index=z["scenario_id"])
    s = val_result.scores.set_index("scenario_id")["anomaly_score"]
    met = u.metrics_at(s[y == 0].to_numpy(), s[y == 1].to_numpy(), THRESHOLD)
    saved = json.loads(config.LOCKED_SELECTION_JSON.read_text())["validation_metrics"]
    for k in ("accuracy", "precision", "recall", "f1", "auc", "fpr"):
        assert met[k] == pytest.approx(saved[k], abs=1e-9), k
    cm = pd.read_csv(STAGE5_DIR / "validation" / "validation" / "confusion_matrix.csv", index_col=0).to_numpy()
    assert [met["tn"], met["fp"], met["fn"], met["tp"]] == cm.flatten().tolist()


def test_locked_test_scores_match_saved_predictions(detector):
    tp = pd.read_csv(STAGE5_DIR / "testing" / "testing" / "scenario_predictions.csv").set_index("scenario_id")
    s = pd.concat([detector.predict_arrays(inf.batch_from_npz(SPLIT_NPZ[k])).scores for k in ("test_normal", "test_attack")])
    s = s.set_index("scenario_id").loc[tp.index]
    assert (s["anomaly_score"] - tp["anomaly_score"]).abs().max() < TOL
    assert (s["is_suspicious"].to_numpy() == tp["predicted_attack"].astype(bool).to_numpy()).all()
    saved = pd.read_csv(STAGE5_DIR / "testing" / "testing" / "locked_test_results.csv").iloc[0]
    with np.load(SPLIT_NPZ["test_normal"]) as zn, np.load(SPLIT_NPZ["test_attack"]) as za:
        neg = s.loc[zn["scenario_id"], "anomaly_score"].to_numpy()
        pos = s.loc[za["scenario_id"], "anomaly_score"].to_numpy()
    met = u.metrics_at(neg, pos, THRESHOLD)
    assert (met["tp"], met["fn"], met["fp"], met["tn"]) == (954, 490, 49, 1063)
    assert met["f1"] == pytest.approx(float(saved["f1"]), abs=1e-9) and met["auc"] == pytest.approx(float(saved["auc"]), abs=1e-9)


# =========================================================================== end-to-end with the previous stages
def test_end_to_end_stage2_to_inference_on_real_graphs(detector, graphs, val_ref):
    r = detector.predict_collection(graphs)
    s = r.scores
    assert len(s) == 12780 and list(s.columns) == ["scenario_id", "anomaly_score", "threshold", "prediction",
                                                   "is_suspicious", "n_nodes", "n_edges"]
    assert np.array_equal(s["scenario_id"], graphs.scenario_ids) and np.isfinite(s["anomaly_score"]).all()
    assert set(s["prediction"]) <= {"Normal", "Suspicious"} and s["prediction"].nunique() == 2     # real, not forced
    sv = s.set_index("scenario_id").loc[val_ref.index]
    assert (sv["anomaly_score"] - val_ref["knn_embedding"]).abs().max() < TOL
    assert (sv["is_suspicious"].to_numpy() == val_ref["predicted_attack"].astype(bool).to_numpy()).all()


def test_scenario_and_event_references_are_kept_for_handoff(detector, graphs):
    r = detector.predict_collection(graphs, graphs.scenario_ids[[10, 20]])
    sid = int(r.scores["scenario_id"].iloc[0])
    ev = r.events_of(sid)
    n = int(r.scores["n_nodes"].iloc[0])
    assert len(ev) == n and (ev["scenario_id"] == sid).all() and ev["node"].is_unique
    assert (ev["node"] == ev["kind"] + ":" + ev["row"].astype(str)).all()
    assert (ev["node_error"] >= 0).all() and len(r.edge_error) == len(r.graphs.edge_attr)


def test_label_free_subset_matches_full_run(detector, graphs):
    full = detector.predict_collection(graphs).scores.set_index("scenario_id")
    ids = graphs.scenario_ids[::997][:10]
    sub = detector.predict_collection(graphs, ids).scores.set_index("scenario_id")
    assert list(sub.index) == sorted(ids.tolist())
    assert (sub["anomaly_score"] - full.loc[sub.index, "anomaly_score"]).abs().max() < 1e-6
    assert (sub["prediction"] == full.loc[sub.index, "prediction"]).all()


def test_unknown_scenario_id_is_rejected(detector, graphs):
    with pytest.raises(KeyError):
        detector.predict_collection(graphs, [10 ** 9])
