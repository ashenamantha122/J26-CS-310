"""Run Stage 2 -> 3 -> 3b on the real CSVs and compare with the saved Stage 3b references.
Writes only to verification/stage3b/ (never to the notebook output folders)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xdr_c3 import scenario_grouping as sg, stage2_validate as s2  # noqa: E402

REF = ROOT / "Stages" / "Stage 3b" / "Outputs"
OUT = ROOT / "verification" / "stage3b"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
data = s2.load_and_validate()
res = sg.run_stage3b(data)
assign_eval, summary_eval = sg.evaluation_tables(res, data)
secs = round(time.time() - t0, 1)

ref_links = pd.read_csv(REF / "links_stage3b_v6v2.csv")
ref_assign = pd.read_csv(REF / "scenario_assignments_v6v2.csv")
ref_summary = pd.read_csv(REF / "scenario_summary_v6v2.csv", keep_default_na=False)  # empty hosts stay ""


def write(df, name):
    p = OUT / name
    df.to_csv(p, index=False, lineterminator="\n")
    return p


def same_file(df, name, ref_name):
    p = write(df, name)
    return p.read_bytes() == (REF / ref_name).read_bytes()


def frame_equal(a, b):
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        return True
    except AssertionError as e:
        return str(e).splitlines()[0]


r = {"seconds": secs}
# ---- links
r["links"] = {"generated": len(res.links), "reference": len(ref_links), "expected_99095": len(res.links) == 99095,
              "types_equal": res.links.link_type.value_counts().sort_index().to_dict()
                             == ref_links.link_type.value_counts().sort_index().to_dict(),
              "type_counts": res.links.link_type.value_counts().sort_index().to_dict(),
              "frame_equal_original_order": frame_equal(res.links.reset_index(drop=True), ref_links),
              "bytes_identical": same_file(res.links, "links_stage3b_regenerated.csv", "links_stage3b_v6v2.csv")}
# ---- assignments
r["assignments"] = {
    "generated": len(res.assignments), "reference": len(ref_assign), "events_total": len(res.nodes),
    "structural_equal_node_scenario_id": frame_equal(res.assignments, ref_assign[["node", "scenario_id"]]),
    "labelled_equal_incl_type": frame_equal(assign_eval, ref_assign),
    "labelled_bytes_identical": same_file(assign_eval, "scenario_assignments_regenerated.csv",
                                          "scenario_assignments_v6v2.csv")}
# ---- summary
struct_ref = ref_summary[sg.SUMMARY_STRUCTURAL_COLUMNS]
r["summary"] = {
    "generated": len(res.summary), "reference": len(ref_summary),
    "structural_columns_equal": frame_equal(res.summary.reset_index(drop=True), struct_ref),
    "labelled_equal_all_columns": frame_equal(summary_eval.reset_index(drop=True), ref_summary),
    "labelled_bytes_identical": same_file(summary_eval, "scenario_summary_regenerated.csv", "scenario_summary_v6v2.csv"),
    "label_columns_equal_separately": {c: frame_equal(summary_eval[[c]].reset_index(drop=True), ref_summary[[c]])
                                       for c in sg.SUMMARY_LABEL_COLUMNS}}
# ---- scenario structure
size = np.bincount(res.scenario)
inc_ids = res.summary["scenario_id"].to_numpy()
r["scenarios"] = {
    "all_scenarios_incl_single_events": int(len(size)), "single_event": int((size == 1).sum()),
    "with_2plus_events": int((size >= 2).sum()), "events_in_2plus": int(size[size >= 2].sum()),
    "min_size_2plus": int(size[size >= 2].min()), "max_size": int(size.max()),
    "median_size_2plus": float(np.median(size[size >= 2])),
    "size_value_counts_head": {int(k): int(v) for k, v in
                               pd.Series(size[size >= 2]).value_counts().sort_index().head(6).items()},
    "scenario_ids_equal_reference": bool(np.array_equal(inc_ids, ref_summary["scenario_id"].to_numpy())),
    "per_node_ids_equal_reference": bool(np.array_equal(res.scenario, ref_assign["scenario_id"].to_numpy())),
    "type_counts": summary_eval["type"].value_counts().to_dict(),
    "event_coverage_all_events_once": bool(res.assignments["node"].is_unique and len(res.assignments) == len(data.host) + len(data.iot)),
    "summary_n_events_sum": int(res.summary["n_events"].sum()),
    "summary_matches_assignment_counts": bool(
        (pd.Series(res.scenario).value_counts().reindex(inc_ids).to_numpy() == res.summary["n_events"].to_numpy()).all())}
# 15-minute continuity: consecutive events (time order) inside any scenario are never > 900 s apart
order = np.lexsort((res.nodes["ts_s"].to_numpy(), res.scenario))
sc, ts = res.scenario[order], res.nodes["ts_s"].to_numpy()[order]
same = sc[1:] == sc[:-1]
r["scenarios"]["max_consecutive_gap_inside_scenario_s"] = int((ts[1:] - ts[:-1])[same].max())
r["scenarios"]["continuity_15min_holds"] = r["scenarios"]["max_consecutive_gap_inside_scenario_s"] <= sg.GATE_GAP
(OUT / "comparison.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r, indent=2))
