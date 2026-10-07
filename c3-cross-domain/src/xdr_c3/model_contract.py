"""Model input contract for the saved R-GCN-GAE: feature order, scaler and relation vocabulary.

The saved research artifacts are the single source of truth; nothing here is derived from incoming data:
  * `feature_names_v6v2 (1).csv`  - the 92 node features, their groups and which ones were scaled
  * `scaler_b1_v6v2.csv`          - training-fitted mean / std (never refitted)
  * the selected checkpoint's stored config (feature columns `cols`, relation names `rel_names`, `rel_cols`),
    read WITHOUT torch through a restricted unpickler (the file is only read, never modified)
  * `training/configuration.json` (84 input features, 10 relations)
"""
from __future__ import annotations

import collections
import json
import pickle
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

EDGE_BASE_FEATURES = ["e_score", "e_log_gap", "e_same_host", "e_short_gap"]
EDGE_SCALED = ["e_score", "e_log_gap"]


class ContractError(ValueError):
    """The saved artifacts disagree with each other."""


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Stub()


class _ConfigUnpickler(pickle.Unpickler):
    """Reads a torch checkpoint's pickle, stubbing every torch object; only OrderedDict is allowed besides."""

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module.startswith("torch"):
            return _Stub()
        raise pickle.UnpicklingError(f"unexpected global in checkpoint: {module}.{name}")

    def persistent_load(self, pid):
        return _Stub()


def read_checkpoint_config(path=config.CHECKPOINT_PATH):
    """Return the config dict stored inside a checkpoint (kind, f_in, e_dim, cols, rel_cols, rel_names, epoch, ...)."""
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        with z.open(member) as fh:
            obj = _ConfigUnpickler(fh).load()
    return dict(obj["config"])


@dataclass(frozen=True)
class ModelContract:
    feature_names: tuple            # 92 node feature names, saved order
    feature_groups: tuple           # group of each feature ('component' ones are not model inputs)
    node_scaler: dict               # name -> (mean_train, std_train)
    edge_scaler: dict
    model_cols: tuple               # indices (into the 92) of the 84 model inputs
    edge_feature_names: tuple       # 14: 4 base + 10 relations
    relation_names: tuple           # 10 'e_type=...' names, saved order
    relation_cols: tuple            # their indices in the 14 edge features
    checkpoint_epoch: int

    @property
    def relation_link_types(self):
        return tuple(r[len("e_type="):] for r in self.relation_names)

    @property
    def scaled_node_features(self):
        return tuple(self.node_scaler)


def load_contract(feature_names_csv=config.FEATURE_NAMES_CSV, scaler_csv=config.SCALER_CSV,
                  checkpoint=config.CHECKPOINT_PATH, model_config_json=config.MODEL_CONFIG_JSON) -> ModelContract:
    fn = pd.read_csv(feature_names_csv)
    scaler = pd.read_csv(scaler_csv)
    ck = read_checkpoint_config(checkpoint)
    mc = json.loads(open(model_config_json, encoding="utf-8").read())

    names, groups = tuple(fn["feature"]), tuple(fn["group"])
    node_scaler = {r.column: (float(r.mean_train), float(r.std_train)) for r in scaler[scaler.table == "node"].itertuples()}
    edge_scaler = {r.column: (float(r.mean_train), float(r.std_train)) for r in scaler[scaler.table == "edge"].itertuples()}

    problems = []
    if len(names) != 92:
        problems.append(f"expected 92 node features, found {len(names)}")
    if set(node_scaler) != {n for n, s in zip(names, fn["scaled_later"]) if s}:
        problems.append("scaler node columns differ from the 'scaled_later' features")
    if tuple(edge_scaler) != tuple(EDGE_SCALED):
        problems.append(f"scaler edge columns {tuple(edge_scaler)} != {tuple(EDGE_SCALED)}")
    model_cols = tuple(int(c) for c in ck["cols"])
    if model_cols != tuple(i for i, g in enumerate(groups) if g != "component"):
        problems.append("checkpoint cols != non-component feature indices")
    if len(model_cols) != ck["f_in"] or ck["f_in"] != mc["input_features"] or ck["f_in"] != 84:
        problems.append("input feature count disagrees between checkpoint and configuration.json")
    rel_names = tuple(ck["rel_names"])
    if list(rel_names) != list(mc["relations"]) or len(rel_names) != 10:
        problems.append("relation names differ between checkpoint and configuration.json")
    edge_names = tuple(EDGE_BASE_FEATURES) + rel_names
    if len(edge_names) != ck["e_dim"] or tuple(ck["rel_cols"]) != tuple(range(4, 14)):
        problems.append("edge feature layout disagrees with the checkpoint (e_dim / rel_cols)")
    if problems:
        raise ContractError("; ".join(problems))
    return ModelContract(names, groups, node_scaler, edge_scaler, model_cols, edge_names, rel_names,
                         tuple(ck["rel_cols"]), int(ck["epoch"]))
