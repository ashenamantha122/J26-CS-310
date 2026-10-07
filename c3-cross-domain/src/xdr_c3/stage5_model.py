"""Stage 5 - PyTorch part: RGCN graph autoencoder (R-GCN-GAE), plain PyTorch (no PyTorch Geometric).

Normal-only training: the model learns to rebuild normal scenario graphs (node features, edge features and which edges
exist). A scenario that is rebuilt badly is anomalous. Per-node and per-edge errors give the suspicious path.

Other encoders (R-GCN-GAE, R-GCN-GAE) are added later by extending `build_model`; everything else is shared.
"""
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import stage5_utils as u   # packaging change only (was: import stage5_utils as u)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_torch(batch, device):
    t = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in batch.items() if isinstance(v, np.ndarray)}
    t["n_graphs"] = batch["n_graphs"]
    return t


# --------------------------------------------------------------------------- model
class RGCNLayer(nn.Module):
    """Relational GCN with a separate message transform for every edge type.

    Aggregation is mean per (destination, relation), followed by sum over relations.
    This avoids treating every edge as the same kind of connection.
    """
    def __init__(self, d_in, d_out, n_rel):
        super().__init__()
        self.rel_weight = nn.Parameter(torch.empty(n_rel, d_in, d_out))
        self.self_linear = nn.Linear(d_in, d_out)
        nn.init.xavier_uniform_(self.rel_weight)

    def forward(self, h, edge_index, relation_ids):
        src, dst = edge_index
        out = self.self_linear(h)
        for r in range(self.rel_weight.size(0)):
            mask = relation_ids == r
            if not torch.any(mask):
                continue
            dest = dst[mask]
            msgs = h[src[mask]] @ self.rel_weight[r]
            accum = torch.zeros_like(out).index_add_(0, dest, msgs)
            deg = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype).index_add_(
                0, dest, torch.ones(dest.shape[0], device=h.device, dtype=h.dtype))
            out = out + accum / deg.clamp(min=1).unsqueeze(-1)
        return out


class RGCNGAE(nn.Module):
    """Two relational GCN encoder layers; same reconstruction decoders as other baselines."""
    def __init__(self, f_in, e_dim, hidden=64, latent=32, dropout=0.1, rel_cols=()):
        super().__init__()
        if len(rel_cols) < 2:
            raise ValueError('R-GCN requires at least two e_type= relation columns')
        self.f_in, self.e_dim = f_in, e_dim
        self.register_buffer('rel_cols', torch.tensor(list(rel_cols), dtype=torch.long))
        self.n_relations = len(rel_cols)
        self.l1 = RGCNLayer(f_in, hidden, self.n_relations)
        self.l2 = RGCNLayer(hidden, latent, self.n_relations)
        self.drop = nn.Dropout(dropout)
        self.dec_node = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, f_in))
        self.dec_edge = nn.Sequential(nn.Linear(3 * latent, hidden), nn.ReLU(), nn.Linear(hidden, e_dim + 1))

    def encode(self, x, edge_index, edge_attr):
        relation_ids = torch.argmax(edge_attr[:, self.rel_cols], dim=1)
        h = self.drop(F.relu(self.l1(x, edge_index, relation_ids)))
        return self.l2(h, edge_index, relation_ids)

    def edge_out(self, z, src, dst):
        zz = torch.cat([z[src], z[dst], (z[src] - z[dst]).abs()], dim=1)
        return self.dec_edge(zz)


def build_model(kind, f_in, e_dim, hidden=64, latent=32, dropout=0.1, rel_cols=()):
    if kind == "rgcn":
        return RGCNGAE(f_in, e_dim, hidden, latent, dropout, rel_cols)
    raise NotImplementedError(f"Unsupported encoder: {kind}")


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- forward pass and loss
def run_model(model, tb, gen=None):
    """Returns the embeddings and, per node / per edge, the reconstruction errors."""
    x, ei, ea = tb["x"], tb["edge_index"], tb["edge_attr"]
    z = model.encode(x, ei, ea)
    node_err = ((model.dec_node(z) - x) ** 2).mean(dim=1)
    src, dst = ei[0], ei[1]
    out = model.edge_out(z, src, dst)
    attr_err = ((out[:, :-1] - ea) ** 2).mean(dim=1)
    pos = F.softplus(-out[:, -1])                              # -log sigmoid(logit): the edge should exist
    # Sample ONLY non-existing directed edges within each graph. Never sample self-loops.
    # Some tiny/dense graphs have zero legal negatives; handle them safely.
    src_cpu = src.detach().cpu().tolist()
    dst_cpu = dst.detach().cpu().tolist()
    starts_cpu = tb["node_start"].detach().cpu().tolist()
    counts_cpu = tb["node_count"].detach().cpu().tolist()
    existing = set(zip(src_cpu, dst_cpu))
    neg_src_parts, neg_dst_parts = [], []
    graph_starts = sorted(set(starts_cpu))
    for start in graph_starts:
        count = counts_cpu[start]
        graph_end = start + count
        n_positive = sum(start <= a < graph_end for a in src_cpu)
        if n_positive == 0:
            continue
        available = [(a, b) for a in range(start, graph_end)
                             for b in range(start, graph_end)
                             if a != b and (a, b) not in existing]
        if not available:
            continue
        # Sample about as many negative pairs as positive edges in this graph.
        draw = torch.randint(len(available), (n_positive,), device=x.device, generator=gen)
        chosen = [available[j] for j in draw.cpu().tolist()]
        neg_src_parts.extend(a for a, b in chosen)
        neg_dst_parts.extend(b for a, b in chosen)
    if neg_src_parts:
        neg_src = torch.tensor(neg_src_parts, dtype=torch.long, device=x.device)
        neg_dst = torch.tensor(neg_dst_parts, dtype=torch.long, device=x.device)
        neg = F.softplus(model.edge_out(z, neg_src, neg_dst)[:, -1])
    else:
        # Zero-valued differentiable placeholder when all graphs are complete.
        neg = z.sum().reshape(1) * 0.0
    return z, node_err, attr_err, pos, neg


def loss_fn(node_err, attr_err, pos, neg, w_edge=1.0, w_exist=0.5):
    return node_err.mean() + w_edge * attr_err.mean() + w_exist * (pos.mean() + neg.mean())


@torch.no_grad()
def eval_loss(model, b, cols, device, batch_size=256, max_graphs=None, seed=2024):
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    n = b.n if max_graphs is None else min(max_graphs, b.n)
    tot = 0.0
    for s in range(0, n, batch_size):
        idx = np.arange(s, min(s + batch_size, n))
        tb = to_torch(u.make_batch_np(b, idx, cols), device)
        _, ne, ae, p, ng = run_model(model, tb, gen)
        tot += float(loss_fn(ne, ae, p, ng)) * len(idx)
    return tot / n


# --------------------------------------------------------------------------- training (normal graphs only, no validation data)
def train_epochs(model, train_b, cols, device, epochs, lr, wd, batch_size, seed, save_dir, config, log=print, max_graphs=None):
    """Pure training: only the training graphs are used. The model is saved after EVERY epoch, so the validation notebook
    can later choose the best epoch without the training notebook ever seeing validation data."""
    set_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_tr = train_b.n if max_graphs is None else min(max_graphs, train_b.n)
    rng = np.random.default_rng(seed)
    os.makedirs(save_dir, exist_ok=True)
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = rng.permutation(n_tr)
        tot = 0.0
        for s in range(0, n_tr, batch_size):
            idx = perm[s:s + batch_size]
            tb = to_torch(u.make_batch_np(train_b, idx, cols), device)
            _, ne, ae, p, ng = run_model(model, tb)
            loss = loss_fn(ne, ae, p, ng)
            if not torch.isfinite(loss):
                raise RuntimeError("the loss became NaN / inf - lower the learning rate or check the features")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(idx)
        hist.append({"epoch": ep, "train_loss": tot / n_tr, "seconds": time.perf_counter() - t0})
        save_checkpoint(model, os.path.join(save_dir, f"epoch_{ep:03d}.pt"), {**config, "epoch": ep})
        log(f"epoch {ep:3d} | training loss {hist[-1]['train_loss']:.4f} | {hist[-1]['seconds']:.1f}s | saved epoch_{ep:03d}.pt")
    return pd.DataFrame(hist)


# --------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_bucket(model, b, cols, device, batch_size=256, seed=123):
    """Scores EVERY graph of a bucket. Returns graph-level errors, mean embeddings and the node / edge errors aligned
    with the bucket arrays (needed for the suspicious-path extraction)."""
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rows, embs, ne_all, ee_all = [], [], [], []
    t0 = time.perf_counter()
    for s in range(0, b.n, batch_size):
        idx = np.arange(s, min(s + batch_size, b.n))
        bn = u.make_batch_np(b, idx, cols)
        tb = to_torch(bn, device)
        z, ne, ae, p, _ = run_model(model, tb, gen)
        ne, ee, zz = ne.cpu().numpy(), (ae + p).cpu().numpy(), z.cpu().numpy()
        cnt = bn["counts"]
        st = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        ecnt = b.n_edges[idx]
        est = np.concatenate([[0], np.cumsum(ecnt)[:-1]])
        rows.append(np.stack([np.add.reduceat(ne, st) / cnt, np.maximum.reduceat(ne, st),
                              np.add.reduceat(ee, est) / ecnt, np.maximum.reduceat(ee, est)], axis=1))
        embs.append(np.add.reduceat(zz, st, axis=0) / cnt[:, None])
        ne_all.append(ne)
        ee_all.append(ee)
    seconds = time.perf_counter() - t0
    graph = pd.DataFrame(np.concatenate(rows), columns=["node_mean", "node_max", "edge_mean", "edge_max"])
    graph.insert(0, "recon_total", graph["node_mean"] + graph["edge_mean"])      # main score: the training objective itself
    return {"graph": graph, "emb": np.concatenate(embs), "node_err": np.concatenate(ne_all),
            "edge_err": np.concatenate(ee_all), "seconds_per_graph": seconds / b.n}


def save_checkpoint(model, path, config):
    """config must hold plain Python numbers / strings / lists (it is saved inside the file)."""
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_checkpoint(path, device):
    """Rebuild a saved model from its file; returns (model in evaluation mode, config)."""
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    model = build_model(cfg["kind"], cfg["f_in"], cfg["e_dim"], cfg["hidden"], cfg["latent"], cfg["dropout"], cfg["rel_cols"])
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    model.eval()
    return model, cfg
