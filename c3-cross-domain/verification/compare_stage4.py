"""Run Stage 2 -> 3 -> 3b -> 4 on the real CSVs and compare the graphs with the saved Stage 4 outputs.
Writes only to verification/stage4/ (never to the notebook output folders)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xdr_c3 import graph_builder as gb, scenario_grouping as sg, stage2_validate as s2  # noqa: E402
from xdr_c3.config import SPLIT_NPZ, STAGE4_OUT  # noqa: E402
from xdr_c3.model_contract import load_contract  # noqa: E402

OUT = ROOT / "verification" / "stage4"
OUT.mkdir(parents=True, exist_ok=True)
GD = STAGE4_OUT / "Graph Data"
timing = {}
t = time.time(); data = s2.load_and_validate(); timing["stage2_s"] = round(time.time() - t, 1)
t = time.time(); s3b = sg.run_stage3b(data); timing["stage3_3b_s"] = round(time.time() - t, 1)
contract = load_contract()
t = time.time(); gc = gb.build_graphs_from_stage3b(data, s3b, contract); timing["stage4_build_s"] = round(time.time() - t, 1)
node_label, attack = gb.ground_truth_arrays(gc, data)          # evaluation only

res = {"timing": timing, "graphs": gc.n_graphs, "nodes": int(len(gc.nodes)), "logical_edges": int(len(gc.es)),
       "directed_edges": int(2 * len(gc.es)), "report_clean": gc.report.clean,
       "node_features": len(gc.feature_names), "model_features": len(contract.model_cols),
       "edge_features": len(gc.edge_feature_names), "relations": len(contract.relation_names)}

# ---- exact array-level comparison with the four saved NPZ files (reconciled by scenario_id)
buckets = {}
all_ids = []
for name, path in SPLIT_NPZ.items():
    z = np.load(path)
    ids = z["scenario_id"]
    all_ids.extend(ids.tolist())
    a = gc.arrays(ids, node_label, attack)
    r = {"graphs": int(len(ids)), "keys_equal": sorted(a) == sorted(z.files)}
    for k in z.files:
        ref, got = z[k], a[k]
        if got.dtype.kind in "US":
            r[k] = {"equal": bool((got == ref).all())}
        else:
            r[k] = {"dtype_equal": got.dtype == ref.dtype, "shape_equal": got.shape == ref.shape,
                    "exactly_equal": bool(np.array_equal(got, ref)),
                    "max_abs_diff": float(np.abs(got.astype(np.float64) - ref.astype(np.float64)).max())}
    r["graph_order_in_file_ascending_by_scenario_id"] = bool(np.array_equal(ids, np.sort(ids)))
    buckets[name] = r
res["npz"] = buckets
res["buckets_partition_all_graphs"] = bool(sorted(all_ids) == gc.scenario_ids.tolist())

# ---- saved CSV references (node membership/order, edge pairs, rounded features)
nl = pd.read_csv(GD / "final_node_labels_v6v2 (1).csv", usecols=["node_id", "node", "scenario_id"])
res["csv_node_membership_and_order_equal"] = bool(
    nl["node_id"].equals(gc.nodes["node_id"]) and nl["node"].equals(gc.nodes["node"])
    and nl["scenario_id"].equals(gc.nodes["scenario_id"]))
fe = pd.read_csv(GD / "final_edges_v6v2.csv")
offs = np.repeat(gc.node_ptr[:-1], np.diff(gc.edge_ptr_logical))
res["csv_edges"] = {"rows": len(fe), "equal_scenario_and_endpoints": bool(
    np.array_equal(fe["scenario_id"], gc.edge_scenario) and np.array_equal(fe["node_a_id"], gc.lo + offs)
    and np.array_equal(fe["node_b_id"], gc.hi + offs))}
sc_n = pd.read_csv(GD / "final_node_features_v6v2.csv")
contract_names = list(contract.feature_names)
raw = sc_n[contract_names].to_numpy(np.float64)                  # unscaled, rounded to 6 decimals in the CSV
exp = raw.copy()
for c, (mu, sd) in contract.node_scaler.items():
    j = contract_names.index(c)
    exp[:, j] = (raw[:, j] - mu) / sd
d = np.abs(gc.x.astype(np.float64) - exp)
res["csv_node_features_rounding_tolerance"] = {"max_abs_diff_scaled": float(d.max()),
                                               "note": "CSV holds unscaled values rounded to 6 decimals; the NPZ comparison above is exact"}
(OUT / "comparison.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
