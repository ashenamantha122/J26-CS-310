"""Stage 5 - shared helpers (NumPy / pandas / SciPy only, no PyTorch).

Used by every Stage 5 model notebook (GraphSAGE-GAE, R-GCN-GAE, R-GCN-GAE) so the data loading, scoring,
thresholds, metrics and suspicious-path extraction are identical for all models.

The ground-truth label is NEVER an input of a model. `is_attack` and `node_label` are read only to build the
validation / test sets and to evaluate.
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BUCKETS = ("train", "validation", "test_normal", "test_attack")


# --------------------------------------------------------------------------- data
class Bucket:
    """All graphs of one Stage 4 bucket file, kept as flat arrays (one npz = many graphs)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files}                     # read every array once
        self.x = d["x"].astype(np.float32)
        self.node_ptr = d["node_ptr"].astype(np.int64)
        self.edge_index = d["edge_index"].astype(np.int64)
        self.edge_attr = d["edge_attr"].astype(np.float32)
        self.edge_ptr = d["edge_ptr"].astype(np.int64)
        self.scenario_id = d["scenario_id"].astype(np.int64)
        self.is_attack = d["scenario_is_attack"].astype(np.int64)
        self.node_label = d["node_label"].astype(np.int64)
        self.feature_names = [str(s) for s in d["feature_names"]]
        self.feature_groups = [str(s) for s in d["feature_groups"]]
        self.edge_feature_names = [str(s) for s in d["edge_feature_names"]]
        self.n = len(self.scenario_id)
        self.n_nodes = np.diff(self.node_ptr)
        self.n_edges = np.diff(self.edge_ptr)              # directed edges (both directions stored)


def feature_columns(bucket, use_component=False, drop_prefixes=()):
    """Indices of the node features a model may use. Component 1 / 2 outputs are a separate group."""
    cols = []
    for i, (n, g) in enumerate(zip(bucket.feature_names, bucket.feature_groups)):
        if g == "component" and not use_component:
            continue
        if any(n.startswith(p) for p in drop_prefixes):
            continue
        cols.append(i)
    return np.array(cols, dtype=np.int64)


def make_batch_np(b, idx, cols):
    """Join graphs idx[0], idx[1], ... into one disjoint graph (node numbers shifted)."""
    xs, eis, eas, bvec, counts = [], [], [], [], []
    off = 0
    for k, i in enumerate(idx):
        n0, n1 = b.node_ptr[i], b.node_ptr[i + 1]
        e0, e1 = b.edge_ptr[i], b.edge_ptr[i + 1]
        xs.append(b.x[n0:n1][:, cols])
        eis.append(b.edge_index[:, e0:e1] + off)
        eas.append(b.edge_attr[e0:e1])
        bvec.append(np.full(n1 - n0, k, dtype=np.int64))
        counts.append(n1 - n0)
        off += n1 - n0
    counts = np.array(counts, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    bv = np.concatenate(bvec)
    return {"x": np.concatenate(xs), "edge_index": np.concatenate(eis, axis=1), "edge_attr": np.concatenate(eas, axis=0),
            "batch": bv, "node_start": starts[bv], "node_count": counts[bv], "n_graphs": len(idx), "counts": counts}


# --------------------------------------------------------------------------- metrics
def auc(neg, pos):
    """ROC-AUC with 'higher score = more anomalous'; neg = normal scores, pos = attack scores."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    r = pd.Series(np.concatenate([neg, pos])).rank(method="average").to_numpy()
    n0, n1 = len(neg), len(pos)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def metrics_at(neg, pos, thr):
    """Decision rule: score > thr  ->  ANOMALOUS."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    tp, fn = int((pos > thr).sum()), int((pos <= thr).sum())
    fp, tn = int((neg > thr).sum()), int((neg <= thr).sum())
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp and precision == precision else 0.0
    acc = (tp + tn) / (tp + fn + fp + tn)
    return {"threshold": float(thr), "recall": recall, "fpr": fpr, "precision": precision, "f1": f1, "accuracy": acc,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn, "auc": auc(neg, pos)}


def threshold_from_normal(normal_scores, percentile=95.0):
    """Threshold set ONLY from held-out normal graphs: the score below which `percentile` % of them fall."""
    return float(np.percentile(np.asarray(normal_scores, float), percentile))


# --------------------------------------------------------------------------- nearest-neighbour scoring
class KNNScorer:
    """Anomaly score = mean distance to the k nearest NORMAL training embeddings (standardised)."""

    def __init__(self, train_emb, k=5):
        self.mu = train_emb.mean(0)
        self.sd = train_emb.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        self.tree = cKDTree((train_emb - self.mu) / self.sd)
        self.k = k

    def score(self, emb):
        d, _ = self.tree.query((emb - self.mu) / self.sd, k=self.k)
        return d.mean(axis=1)


def bag_of_events(b, cols):
    """Graph summary without any structure: mean and max of the node features (a simple baseline)."""
    out = np.zeros((b.n, 2 * len(cols)), dtype=np.float32)
    for i in range(b.n):
        x = b.x[b.node_ptr[i]:b.node_ptr[i + 1]][:, cols]
        out[i] = np.concatenate([x.mean(0), x.max(0)])
    return out


# --------------------------------------------------------------------------- suspicious path
def extract_path(b, i, node_err, edge_err, node_thr, edge_thr, max_nodes=12):
    """For graph i: take the nodes / edges whose error is above the thresholds (set from normal graphs),
    join them into connected groups, and return the group with the largest total error as the suspicious path
    (nodes in time order, because Stage 4 stores nodes in time order)."""
    n0, n1 = b.node_ptr[i], b.node_ptr[i + 1]
    e0, e1 = b.edge_ptr[i], b.edge_ptr[i + 1]
    n = n1 - n0
    ne = node_err[n0:n1]
    ei = b.edge_index[:, e0:e1]
    half = (e1 - e0) // 2                                   # first half = one direction of every edge
    pair_err = 0.5 * (edge_err[e0:e0 + half] + edge_err[e0 + half:e1])
    hi_nodes = np.where(ne > node_thr)[0]
    hi_edges = np.where(pair_err > edge_thr)[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k in hi_edges:
        a, c = find(int(ei[0, k])), find(int(ei[1, k]))
        if a != c:
            parent[a] = c
    members = {}
    for v in set(hi_nodes.tolist()) | set(ei[0, hi_edges].tolist()) | set(ei[1, hi_edges].tolist()):
        members.setdefault(find(int(v)), []).append(int(v))
    if not members:
        top = int(np.argmax(ne))
        members = {top: [top]}
    best = max(members.values(), key=lambda m: float(ne[m].sum()))
    best = sorted(best)[:max_nodes]
    in_path = set(best)
    types = [c for c in b.edge_feature_names if c.startswith("e_type=")]
    t_idx = [b.edge_feature_names.index(c) for c in types]
    path_edge_types = set()
    for k in range(half):
        if int(ei[0, k]) in in_path and int(ei[1, k]) in in_path:
            for name, j in zip(types, t_idx):
                if b.edge_attr[e0 + k, j] > 0.5:
                    path_edge_types.add(name[len("e_type="):])
    return {"path_nodes": best, "high_error_nodes": hi_nodes.tolist(), "high_error_edges": hi_edges.tolist(),
            "path_edge_types": sorted(path_edge_types),
            "cross_host_hop": "connect_accept" in path_edge_types, "cross_domain_link": "iot_host_match" in path_edge_types}


# --------------------------------------------------------------------------- readable events (optional)
def load_event_lookup(nodes_csv, events_csv):
    """Map (scenario_id, local node number) -> readable event text, if the two optional files are available."""
    if nodes_csv is None or events_csv is None:
        return None
    nodes = pd.read_csv(nodes_csv, usecols=["node_id", "node", "scenario_id"])
    ev = pd.read_csv(events_csv, dtype=str, keep_default_na=False)
    has_net = ~ev["src_ip"].isin(["-", ""])
    tup = np.where(has_net, " " + ev["src_ip"] + "->" + ev["dst_ip"] + ":" + ev["dst_port"], "")
    who = np.where(ev["process_name"] != "", ev["process_name"] + " ", "")
    ev["text"] = ev["timestamp"] + " | " + ev["source"] + " | " + ev["host_or_device"] + " | " + who + ev["event_type"] + tup
    text_of = dict(zip(ev["node"], ev["text"]))
    start = nodes.groupby("scenario_id")["node_id"].min()
    node_names = nodes["node"].to_numpy()
    return {"start": start, "node_names": node_names, "text_of": text_of}


def describe_path(lookup, scenario_id, local_nodes):
    if lookup is None:
        return " -> ".join(f"node{k}" for k in local_nodes)
    s = int(lookup["start"][int(scenario_id)])
    return " -> ".join(lookup["text_of"].get(lookup["node_names"][s + k], f"node{k}") for k in local_nodes)
