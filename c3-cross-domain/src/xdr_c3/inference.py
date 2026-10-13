"""Stage 5 - saved R-GCN-GAE inference (application interface).

Uses the ORIGINAL trained network and scoring code (`stage5_model.score_bucket`, `stage5_utils.KNNScorer`) with the
locked selection from `validation/locked_selection.json`: epoch-40 checkpoint, score `knn_embedding`, k = 5,
threshold 2.5825197985552437, decision rule `score > threshold -> suspicious`. Nothing is trained, tuned or
recalculated from labels.

Application-side additions (none change the model or the scoring arithmetic):
  * the checkpoint is loaded with `torch.load(weights_only=True)` after a SHA-256 integrity check;
  * `GraphBatch` is a label-free twin of `stage5_utils.Bucket` (the original loader requires evaluation labels);
  * the normal-reference embeddings (8,946 training graphs) are rebuilt with the saved model and cached with
    hashes/metadata so a stale or incompatible cache is detected;
  * `validate_graph_arrays` checks pointers, dimensions, feature schema and relation encoding before inference.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config
from . import stage5_model as m
from . import stage5_utils as u
from .model_contract import ModelContract, load_contract

NORMAL, SUSPICIOUS = "Normal", "Suspicious"
EXPECTED_LOCK = {"selected_epoch": 40, "score": "knn_embedding", "knn_k": 5, "threshold": 2.5825197985552437}
CACHE_FORMAT = 1


class InferenceInputError(ValueError):
    """The supplied graph arrays are not valid model input."""


class ArtifactMismatchError(RuntimeError):
    """A saved artifact does not match the locked configuration or the checkpoint."""


# --------------------------------------------------------------------------- label-free graph container
@dataclass
class GraphBatch:
    """Attribute-compatible with `stage5_utils.Bucket` for everything scoring uses, without label arrays."""
    x: np.ndarray
    node_ptr: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_ptr: np.ndarray
    scenario_id: np.ndarray
    feature_names: list
    feature_groups: list
    edge_feature_names: list

    def __post_init__(self):
        self.n = len(self.scenario_id)
        self.n_nodes = np.diff(self.node_ptr)
        self.n_edges = np.diff(self.edge_ptr)


_REQUIRED = ("x", "node_ptr", "edge_index", "edge_attr", "edge_ptr", "scenario_id",
             "feature_names", "feature_groups", "edge_feature_names")


def batch_from_arrays(arrays) -> GraphBatch:
    """Build a `GraphBatch` from Stage 4 arrays (`GraphCollection.arrays()` or an NPZ mapping). Label arrays, if
    present, are ignored. Dtypes follow `stage5_utils.Bucket`; values are not modified."""
    missing = [k for k in _REQUIRED if k not in arrays]
    if missing:
        raise InferenceInputError(f"missing graph arrays: {missing}")
    return GraphBatch(
        x=np.asarray(arrays["x"]).astype(np.float32), node_ptr=np.asarray(arrays["node_ptr"]).astype(np.int64),
        edge_index=np.asarray(arrays["edge_index"]).astype(np.int64),
        edge_attr=np.asarray(arrays["edge_attr"]).astype(np.float32),
        edge_ptr=np.asarray(arrays["edge_ptr"]).astype(np.int64),
        scenario_id=np.asarray(arrays["scenario_id"]).astype(np.int64),
        feature_names=[str(s) for s in arrays["feature_names"]],
        feature_groups=[str(s) for s in arrays["feature_groups"]],
        edge_feature_names=[str(s) for s in arrays["edge_feature_names"]])


def batch_from_npz(path) -> GraphBatch:
    """Read ONLY the model-input arrays of a Stage 4 NPZ (the label arrays are never loaded)."""
    with np.load(path, allow_pickle=False) as z:
        return batch_from_arrays({k: z[k] for k in _REQUIRED})


def validate_graph_arrays(b: GraphBatch, contract: ModelContract, cols=None):
    """Reject anything the saved model cannot consume. Raises `InferenceInputError` with every problem found."""
    p = []
    cols = np.asarray(contract.model_cols if cols is None else cols)
    if b.feature_names != list(contract.feature_names):
        p.append("node feature names/order differ from the saved schema")
    if b.feature_groups != list(contract.feature_groups):
        p.append("node feature groups differ from the saved schema")
    if b.edge_feature_names != list(contract.edge_feature_names):
        p.append("edge feature names/order differ from the saved schema")
    if b.x.ndim != 2 or b.x.shape[1] != len(contract.feature_names):
        p.append(f"x must have {len(contract.feature_names)} columns, got shape {b.x.shape}")
    if b.edge_attr.ndim != 2 or b.edge_attr.shape[1] != len(contract.edge_feature_names):
        p.append(f"edge_attr must have {len(contract.edge_feature_names)} columns, got shape {b.edge_attr.shape}")
    if len(cols) == 0 or cols.min() < 0 or (b.x.ndim == 2 and cols.max() >= b.x.shape[1]):
        p.append("model feature indices fall outside the node feature matrix")
    if p:
        raise InferenceInputError("; ".join(p))
    G = b.n
    if G == 0:
        raise InferenceInputError("no graphs supplied")
    if len(set(b.scenario_id.tolist())) != G:
        p.append("duplicate scenario ids")
    for name, ptr, total in (("node_ptr", b.node_ptr, len(b.x)), ("edge_ptr", b.edge_ptr, len(b.edge_attr))):
        if len(ptr) != G + 1 or ptr[0] != 0 or ptr[-1] != total or (np.diff(ptr) <= 0).any():
            p.append(f"{name} must start at 0, strictly increase and end at {total} (one entry per graph + 1)")
    if b.edge_index.shape != (2, len(b.edge_attr)):
        p.append(f"edge_index shape {b.edge_index.shape} does not match {len(b.edge_attr)} edges")
    if not (np.isfinite(b.x).all() and np.isfinite(b.edge_attr).all()):
        p.append("non-finite values in node or edge features")
    if p:
        raise InferenceInputError("; ".join(p))
    if (b.n_edges % 2).any():
        p.append("a graph has an odd number of directed edges (edges must be stored in both directions)")
    nn = np.repeat(b.n_nodes, b.n_edges)
    if (b.edge_index < 0).any() or (b.edge_index >= nn[None, :]).any():
        p.append("edge endpoints fall outside their graph's node range")
    rel = b.edge_attr[:, list(contract.relation_cols)]
    if not (np.isclose(rel, 0, atol=1e-4) | np.isclose(rel, 1, atol=1e-4)).all() or not np.allclose(rel.sum(1), 1, atol=1e-4):
        p.append("relation columns are not exactly one-hot (unsupported or ambiguous relation encoding)")
    if p:
        raise InferenceInputError("; ".join(p))
    return b


# --------------------------------------------------------------------------- locked configuration and checkpoint
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_locked_selection(path=config.LOCKED_SELECTION_JSON):
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    for k, v in EXPECTED_LOCK.items():
        if lock.get(k) != v:
            raise ArtifactMismatchError(f"locked_selection.json {k}={lock.get(k)!r}, expected {v!r}")
    if lock.get("max_validation_fpr") != 0.05:
        raise ArtifactMismatchError("unexpected validation FPR constraint in locked selection")
    return lock


def load_trusted_checkpoint(path=config.CHECKPOINT_PATH, device="cpu", expected_sha256=config.CHECKPOINT_SHA256):
    """Hash-verified, `weights_only=True` load of the saved model; returns (model in eval mode, stored config)."""
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ArtifactMismatchError(f"checkpoint hash {digest} != pinned {expected_sha256}; refusing to load")
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = ck["config"]
    model = m.build_model(cfg["kind"], cfg["f_in"], cfg["e_dim"], cfg["hidden"], cfg["latent"], cfg["dropout"],
                          cfg["rel_cols"])
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg, digest


def check_checkpoint_against_contract(cfg, contract: ModelContract, lock):
    bad = []
    if cfg["kind"] != "rgcn":
        bad.append(f"kind={cfg['kind']}")
    if cfg["epoch"] != lock["selected_epoch"]:
        bad.append(f"epoch={cfg['epoch']}")
    if cfg["f_in"] != 84 or tuple(cfg["cols"]) != contract.model_cols:
        bad.append("model feature indices differ from the saved schema")
    if cfg["e_dim"] != 14 or tuple(cfg["rel_cols"]) != contract.relation_cols or tuple(cfg["rel_names"]) != contract.relation_names:
        bad.append("edge/relation layout differs from the saved schema")
    if (cfg["hidden"], cfg["latent"], cfg["dropout"]) != (64, 32, 0.1) or cfg["setting"] != "nocomp":
        bad.append("architecture hyper-parameters differ from configuration.json")
    if bad:
        raise ArtifactMismatchError("; ".join(bad))


# --------------------------------------------------------------------------- results
@dataclass
class InferenceResult:
    """Scenario-level predictions plus everything the future F1 -> F2 hand-off needs (event references,
    per-node and per-edge reconstruction errors, embeddings). No ground truth is involved."""
    scores: pd.DataFrame           # scenario_id, anomaly_score, threshold, prediction, is_suspicious, n_nodes, n_edges
    graphs: GraphBatch
    embeddings: np.ndarray         # (G, 32) mean node embedding per graph
    node_error: np.ndarray         # aligned with graphs.x rows
    edge_error: np.ndarray         # aligned with graphs.edge_attr rows (both directions stored)
    graph_errors: pd.DataFrame     # recon_total, node_mean, node_max, edge_mean, edge_max per graph
    nodes: pd.DataFrame = None     # Stage 4 node references (node, kind, row, ts_s, host_ip) when supplied

    def events_of(self, scenario_id):
        """Source events (`host:<row>` / `iot:<row>`) of a scenario, in graph node order, with their errors."""
        i = int(np.flatnonzero(self.graphs.scenario_id == scenario_id)[0])
        s, e = self.graphs.node_ptr[i], self.graphs.node_ptr[i + 1]
        if self.nodes is None:
            raise ValueError("no node references were supplied (use predict_collection)")
        out = self.nodes.iloc[s:e].copy()
        out["node_error"] = self.node_error[s:e]
        return out


# --------------------------------------------------------------------------- detector
class Stage5Detector:
    """Saved R-GCN-GAE + kNN scorer over the reconstructed normal-training embeddings."""

    def __init__(self, model, cfg, cols, knn, reference_embeddings, lock, contract, device, checkpoint_sha256):
        self.model, self.cfg, self.cols, self.knn = model, cfg, cols, knn
        self.reference_embeddings = reference_embeddings
        self.threshold = float(lock["threshold"])
        self.k = int(lock["knn_k"])
        self.lock, self.contract, self.device, self.checkpoint_sha256 = lock, contract, device, checkpoint_sha256

    # -- construction
    @classmethod
    def load(cls, device="cpu", use_cache=True, cache_dir=config.CACHE_DIR, batch_size=256):
        contract = load_contract()
        lock = load_locked_selection()
        model, cfg, digest = load_trusted_checkpoint(device=device)
        check_checkpoint_against_contract(cfg, contract, lock)
        cols = np.array(cfg["cols"], dtype=np.int64)
        ref = cls._reference_embeddings(model, cols, device, contract, digest, use_cache, Path(cache_dir), batch_size)
        knn = u.KNNScorer(ref, k=int(lock["knn_k"]))
        return cls(model, cfg, cols, knn, ref, lock, contract, device, digest)

    @staticmethod
    def _cache_meta(digest, train_path):
        return {"cache_format": CACHE_FORMAT, "checkpoint_sha256": digest, "train_npz_sha256": sha256_file(train_path),
                "feature_names_sha256": sha256_file(config.FEATURE_NAMES_CSV), "torch": torch.__version__,
                "numpy": np.__version__, "scoring": "stage5_model.score_bucket mean-node-embedding"}

    @classmethod
    def _reference_embeddings(cls, model, cols, device, contract, digest, use_cache, cache_dir, batch_size):
        train_path = config.SPLIT_NPZ["train"]
        meta = cls._cache_meta(digest, train_path)
        npz_p, json_p = cache_dir / "reference_embeddings.npz", cache_dir / "reference_embeddings.json"
        if use_cache and npz_p.is_file() and json_p.is_file():
            stored = json.loads(json_p.read_text(encoding="utf-8"))
            if {k: stored.get(k) for k in meta} == meta:
                with np.load(npz_p) as z:
                    emb = z["embeddings"]
                if emb.shape == (stored["n_graphs"], 32) and hashlib.sha256(emb.tobytes()).hexdigest() == stored["embeddings_sha256"]:
                    return emb
        batch = batch_from_npz(train_path)
        validate_graph_arrays(batch, contract, cols)
        emb = m.score_bucket(model, batch, cols, torch.device(device), batch_size=batch_size)["emb"]
        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(npz_p, embeddings=emb)
            json_p.write_text(json.dumps({**meta, "n_graphs": int(emb.shape[0]),
                                          "embeddings_sha256": hashlib.sha256(emb.tobytes()).hexdigest()}, indent=2),
                              encoding="utf-8")
        return emb

    # -- inference
    def predict_arrays(self, arrays, nodes=None, batch_size=256) -> InferenceResult:
        """Score graphs given as Stage 4 arrays (labels optional and ignored)."""
        batch = arrays if isinstance(arrays, GraphBatch) else batch_from_arrays(arrays)
        validate_graph_arrays(batch, self.contract, self.cols)
        t0 = time.perf_counter()
        s = m.score_bucket(self.model, batch, self.cols, torch.device(self.device), batch_size=batch_size)
        score = self.knn.score(s["emb"])                        # mean distance to the k nearest normal embeddings
        suspicious = score > self.threshold                     # original decision rule: score > threshold
        scores = pd.DataFrame({"scenario_id": batch.scenario_id, "anomaly_score": score, "threshold": self.threshold,
                               "prediction": np.where(suspicious, SUSPICIOUS, NORMAL), "is_suspicious": suspicious,
                               "n_nodes": batch.n_nodes, "n_edges": batch.n_edges // 2})
        scores.attrs["seconds_per_graph"] = (time.perf_counter() - t0) / batch.n
        return InferenceResult(scores, batch, s["emb"], s["node_err"], s["edge_err"], s["graph"], nodes)

    def predict_collection(self, collection, scenario_ids=None, batch_size=256) -> InferenceResult:
        """Score a Stage 4 `GraphCollection` (all graphs, or the given scenario ids). Labels are never supplied."""
        arrays = collection.arrays(scenario_ids)
        ids = arrays["scenario_id"]
        nodes = collection.nodes[collection.nodes["scenario_id"].isin(ids)].reset_index(drop=True)
        return self.predict_arrays(arrays, nodes=nodes, batch_size=batch_size)
