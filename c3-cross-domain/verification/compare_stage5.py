"""Stage 2 -> 3 -> 3b -> 4 -> saved R-GCN-GAE inference on the real data, compared with the saved Stage 5 results.
Writes only to verification/stage5/ (never to the research artifact folders)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xdr_c3 import graph_builder as gb, inference as inf, scenario_grouping as sg, stage2_validate as s2  # noqa: E402
from xdr_c3 import stage5_utils as u  # noqa: E402
from xdr_c3.config import SPLIT_NPZ, STAGE5_DIR, VALIDATION_SCORES_CSV  # noqa: E402

OUT = ROOT / "verification" / "stage5"
OUT.mkdir(parents=True, exist_ok=True)
TOL = 1e-5            # absolute score tolerance (observed float32 noise is ~3e-7; see tolerance check below)
timing = {}
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc", "fpr", "tp", "fn", "fp", "tn"]

t = time.time()
det = inf.Stage5Detector.load(use_cache=False)
timing["load_and_reference_embeddings_s"] = round(time.time() - t, 1)
res = {"environment": {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "device": det.device,
                       "numpy": np.__version__, "threads": torch.get_num_threads()},
       "checkpoint": {"sha256": det.checkpoint_sha256, "epoch": det.cfg["epoch"], "kind": det.cfg["kind"],
                      "f_in": det.cfg["f_in"], "e_dim": det.cfg["e_dim"], "relations": len(det.cfg["rel_names"]),
                      "n_parameters": int(sum(p.numel() for p in det.model.parameters())),
                      "training_mode": det.model.training},
       "lock": {"score": det.lock["score"], "k": det.k, "threshold": det.threshold},
       "reference_embeddings": {"shape": list(det.reference_embeddings.shape),
                                "dtype": str(det.reference_embeddings.dtype),
                                "finite": bool(np.isfinite(det.reference_embeddings).all())},
       "tolerance_abs": TOL}


def truth_of(path):                                   # evaluation only: used for metrics, never for scoring
    with np.load(path) as z:
        return pd.Series(z["scenario_is_attack"].astype(int), index=z["scenario_id"])


def metrics(scores, y, thr):
    return u.metrics_at(scores[y == 0].to_numpy(), scores[y == 1].to_numpy(), thr)


# ---- validation graphs from the saved NPZ vs per_graph_validation_scores.csv
ref = pd.read_csv(VALIDATION_SCORES_CSV).set_index("scenario_id")
r_val = det.predict_arrays(inf.batch_from_npz(SPLIT_NPZ["validation"]))
s = r_val.scores.set_index("scenario_id")
same_cov = set(s.index) == set(ref.index) and len(s) == len(ref) == 1278
s = s.loc[ref.index]
diff = (s["anomaly_score"] - ref["knn_embedding"]).abs()
dec_saved = ref["predicted_attack"].astype(bool).to_numpy()
y = truth_of(SPLIT_NPZ["validation"]).loc[ref.index]
m_val = metrics(s["anomaly_score"], y, det.threshold)
cm = pd.read_csv(STAGE5_DIR / "validation" / "validation" / "confusion_matrix.csv", index_col=0)
saved_cm = [int(cm.iloc[0, 0]), int(cm.iloc[0, 1]), int(cm.iloc[1, 0]), int(cm.iloc[1, 1])]     # tn fp fn tp
ge = r_val.graph_errors.set_axis(r_val.scores["scenario_id"]).loc[ref.index]
res["validation_npz_vs_saved"] = {
    "graphs": len(s), "same_graph_coverage": bool(same_cov), "max_abs_score_diff": float(diff.max()),
    "mean_abs_score_diff": float(diff.mean()), "scores_within_tolerance": bool((diff <= TOL).all()),
    "decisions_identical": bool((s["is_suspicious"].to_numpy() == dec_saved).all()),
    "decision_disagreements": int((s["is_suspicious"].to_numpy() != dec_saved).sum()),
    "min_abs_distance_to_threshold": float((s["anomaly_score"] - det.threshold).abs().min()),
    "threshold_distance_exceeds_tolerance": bool((s["anomaly_score"] - det.threshold).abs().min() > TOL),
    "other_score_max_abs_diff": {c: float((ge[c] - ref[c]).abs().max())
                                 for c in ["recon_total", "node_mean", "node_max", "edge_mean", "edge_max"]},
    "metrics_recomputed": {k: m_val[k] for k in METRIC_KEYS},
    "saved_confusion_tn_fp_fn_tp": saved_cm,
    "confusion_equal": [m_val["tn"], m_val["fp"], m_val["fn"], m_val["tp"]] == saved_cm,
    "saved_locked_metrics": json.loads(
        (STAGE5_DIR / "validation" / "validation" / "locked_selection.json").read_text())["validation_metrics"]}
sm = res["validation_npz_vs_saved"]["saved_locked_metrics"]
res["validation_npz_vs_saved"]["metrics_match_saved_1e-9"] = all(
    abs(m_val[k] - sm[k]) < 1e-9 for k in ["accuracy", "precision", "recall", "f1", "auc", "fpr"])

# ---- locked test graphs (normal + attack NPZ) vs testing/scenario_predictions.csv
tp = pd.read_csv(STAGE5_DIR / "testing" / "testing" / "scenario_predictions.csv").set_index("scenario_id")
parts = [det.predict_arrays(inf.batch_from_npz(SPLIT_NPZ[k])).scores for k in ("test_normal", "test_attack")]
ts = pd.concat(parts).set_index("scenario_id")
ty = pd.concat([truth_of(SPLIT_NPZ["test_normal"]), truth_of(SPLIT_NPZ["test_attack"])])
ts = ts.loc[tp.index]
dt = (ts["anomaly_score"] - tp["anomaly_score"]).abs()
m_test = metrics(ts["anomaly_score"], ty.loc[tp.index], det.threshold)
saved_test = pd.read_csv(STAGE5_DIR / "testing" / "testing" / "locked_test_results.csv").iloc[0]
res["test_npz_vs_saved"] = {
    "graphs": len(ts), "max_abs_score_diff": float(dt.max()),
    "decisions_identical": bool((ts["is_suspicious"].to_numpy() == tp["predicted_attack"].astype(bool).to_numpy()).all()),
    "labels_in_saved_file_match_npz": bool((ty.loc[tp.index].to_numpy() == tp["true_label"].to_numpy()).all()),
    "recomputed": {k: m_test[k] for k in METRIC_KEYS}, "saved": {k: float(saved_test[k]) for k in METRIC_KEYS},
    "metrics_equal_saved_1e-9": all(abs(m_test[k] - float(saved_test[k])) < 1e-9 for k in METRIC_KEYS)}

# ---- full application pathway: Stage 2 -> 3 -> 3b -> 4 -> inference (label-free)
t = time.time(); data = s2.load_and_validate(); timing["stage2_s"] = round(time.time() - t, 1)
t = time.time(); s3b = sg.run_stage3b(data); timing["stage3_3b_s"] = round(time.time() - t, 1)
t = time.time(); gc = gb.build_graphs_from_stage3b(data, s3b, det.contract); timing["stage4_s"] = round(time.time() - t, 1)
t = time.time(); full = det.predict_collection(gc); timing["inference_all_graphs_s"] = round(time.time() - t, 1)
fs = full.scores.set_index("scenario_id")
vs = fs.loc[ref.index]
res["pipeline_vs_saved_validation"] = {
    "graphs_scored": len(fs), "max_abs_score_diff_vs_saved": float((vs["anomaly_score"] - ref["knn_embedding"]).abs().max()),
    "decisions_identical_vs_saved": bool((vs["is_suspicious"].to_numpy() == dec_saved).all()),
    "scores_identical_to_npz_route": bool(np.array_equal(vs["anomaly_score"].to_numpy(), s["anomaly_score"].to_numpy()))}
tsc = fs.loc[tp.index]
res["pipeline_vs_saved_test"] = {
    "max_abs_score_diff_vs_saved": float((tsc["anomaly_score"] - tp["anomaly_score"]).abs().max()),
    "decisions_identical_vs_saved": bool((tsc["is_suspicious"].to_numpy() == tp["predicted_attack"].astype(bool).to_numpy()).all())}
res["pipeline_prediction_counts"] = fs["prediction"].value_counts().to_dict()
res["score_summary"] = {k: float(v) for k, v in fs["anomaly_score"].describe().items()}

# ---- label-free subset through the application interface
ids = gc.scenario_ids[::1000][:12]
sub = det.predict_collection(gc, ids)
sid = sub.scores["scenario_id"]
res["label_free_subset"] = {
    "scenario_ids": [int(i) for i in sid],
    "matches_full_run_atol_1e-6": bool(np.allclose(sub.scores["anomaly_score"].to_numpy(), fs.loc[sid, "anomaly_score"].to_numpy(), rtol=0, atol=1e-6)),
    "same_decisions": bool((sub.scores["is_suspicious"].to_numpy() == fs.loc[sid, "is_suspicious"].to_numpy()).all())}
res["examples_actual_scores"] = sub.scores.round(6).head(8).to_dict("records")
top = full.scores.sort_values("anomaly_score", ascending=False).head(3)
res["examples_highest_scores"] = top.round(6).to_dict("records")
res["example_event_references_top_scenario"] = full.events_of(int(top.iloc[0]["scenario_id"])).head(5).round(6).to_dict("records")
res["timing"] = timing
full.scores.to_csv(OUT / "scenario_scores_all_graphs.csv", index=False, float_format="%.9g")
(OUT / "comparison.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))
