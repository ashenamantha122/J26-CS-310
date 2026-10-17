"""Function 2 on the real data: Stage 2 -> 3 -> 3b -> 4 -> saved R-GCN inference -> F1->F2 handoff -> F2.
Writes only to verification/f2/. Ground-truth labels are NOT read anywhere in this script.
No F2 precision/recall/accuracy is computed: no independent lateral-movement ground truth exists.

Timing note: F2 is timed WITHOUT tracemalloc (tracing slows this pandas-heavy code about 5x); memory is measured in a
separate pass whose elapsed time is not reported as runtime."""
import collections
import ctypes
import gc
import json
import statistics
import sys
import time
import tracemalloc
from ctypes import wintypes
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xdr_c3 import f1_f2_handoff as h, graph_builder as gb, inference as inf, lateral_movement as lm  # noqa: E402
from xdr_c3 import scenario_grouping as sg, stage2_validate as s2  # noqa: E402

OUT = ROOT / "verification" / "f2"
OUT.mkdir(parents=True, exist_ok=True)


def peak_working_set_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)


timing = {}
t = time.time(); data = s2.load_and_validate(); timing["stage2_s"] = round(time.time() - t, 1)
t = time.time(); s3b = sg.run_stage3b(data); timing["stage3_3b_s"] = round(time.time() - t, 1)
t = time.time(); gc_ = gb.build_graphs_from_stage3b(data, s3b); timing["stage4_s"] = round(time.time() - t, 1)
t = time.time(); det = inf.Stage5Detector.load(); res = det.predict_collection(gc_); timing["f1_inference_s"] = round(time.time() - t, 1)
f1_before = res.scores.copy()
t = time.time(); ho = h.build_f1_f2_handoff(data, s3b, res, det.contract); timing["handoff_s"] = round(time.time() - t, 1)

# ---- gate: records vs authoritative F1 results
gate_problems = lm.check_f1_gate(ho)

# ---- representative records first, then everything (untraced timing, repeated)
t = time.perf_counter()
for sid in list(ho.records)[:10]:
    lm.analyse_scenario(ho.records[sid])
timing["f2_first_10_records_s"] = round(time.perf_counter() - t, 3)
runs = []
for _ in range(3):
    gc.collect()
    t = time.perf_counter(); rep = lm.analyse_handoff(ho); runs.append(time.perf_counter() - t)
timing["f2_all_1499_untraced_runs_s"] = [round(x, 2) for x in runs]
timing["f2_all_1499_untraced_median_s"] = round(statistics.median(runs), 2)
before_mem = peak_working_set_mb()
tracemalloc.start(); lm.analyse_handoff(ho); _, heap_peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
after_mem = peak_working_set_mb()

A = rep.assessments
counts = rep.counts()
host, _ = data.to_stage3_frames()
H = host.set_index("row")
sup = [a for a in A.values() if a.assessment == lm.SUPPORTED]
svc = [a for a in A.values() if a.assessment == lm.SERVICE_CHAIN_ONLY]
flagged = [a for a in A.values() if a.flags]

summary = {
    "counts": counts,
    "reconciles": counts["analysed"] == len(ho.records) == sum(counts[k] for k in (lm.SUPPORTED, lm.SERVICE_CHAIN_ONLY, lm.NO_PATH, lm.INSUFFICIENT, "processing_errors")),
    "reason_codes": dict(collections.Counter(a.reason_code for a in A.values())),
    "processing_errors": rep.errors,
    "f1_gate": {"violations": gate_problems, "f1_scores_unchanged_by_f2": bool(res.scores.equals(f1_before) and ho.f1_scores.equals(f1_before)),
                "assessment_f1_equals_authoritative": all(
                    a.f1["anomaly_score"] == float(ho.f1_scores.set_index("scenario_id").at[sid, "anomaly_score"]) for sid, a in A.items())},
    "search": {"max_candidates_per_scenario": max(a.search["candidates"] for a in A.values()),
               "truncated_scenarios": sum(a.search["truncated"] for a in A.values()),
               "scenarios_with_host_revisits_rejected": sum(a.search["host_revisits_rejected"] > 0 for a in A.values()),
               "scenarios_with_origin_revisits_rejected": sum(a.search["origin_revisits_rejected"] > 0 for a in A.values())},
    "evidence_problems_total": sum(len(a.evidence_problems) for a in A.values()),
    "potential_lateral_movement_supported": {
        "scenarios": len(sup),
        "evidence_types": dict(collections.Counter("+".join(a.evidence_types) for a in sup).most_common()),
        "best_path_hops": dict(collections.Counter(max(p["n_hops"] for p in a.qualifying_paths) for a in sup)),
        "entry_process_of_first_qualifying_path": dict(collections.Counter(a.qualifying_paths[0]["segments"][0]["entry"]["process_name"] for a in sup)),
        "entry_source_class": dict(collections.Counter(a.qualifying_paths[0]["segments"][0]["entry_context"]["source_class"] for a in sup)),
        "first_hop_same_second_share": round(sum(a.qualifying_paths[0]["segments"][0]["hop"]["accept_minus_connect_s"] == 0 for a in sup) / len(sup), 3)},
    "cross_host_service_chain_only": {
        "scenarios": len(svc), "best_path_hops": dict(collections.Counter(max(p["n_hops"] for p in a.supported_paths) for a in svc)),
        "entry_process": dict(collections.Counter(a.supported_paths[0]["segments"][0]["entry"]["process_name"] for a in svc)),
        "destination_ports": dict(collections.Counter(a.supported_paths[0]["segments"][0]["hop"]["destination"]["port"] for a in svc))},
    "destination_side_activity_flagged_no_supporting_path": len(flagged),
    "scenario_36595": {"assessment": A[36595].assessment, "evidence_types": A[36595].evidence_types,
                       "path_classifications": [p["classification"] for p in A[36595].supported_paths]},
    "scenario_22761": {"assessment": A[22761].assessment, "evidence_types": A[22761].evidence_types,
                       "max_hops": max(p["n_hops"] for p in A[22761].supported_paths)},
    "unique_path_ids": len({p["path_id"] for a in A.values() for p in a.paths}) == sum(len(a.paths) for a in A.values()),
    "timing": timing,
    "memory": {"process_peak_working_set_mb_before_f2_memory_pass": before_mem, "process_peak_working_set_mb_after": after_mem,
               "python_heap_peak_during_f2_mb": round(heap_peak / 2 ** 20, 2)},
    "scientific_note": ("potential_lateral_movement_supported means a verified, directed, time-consistent cross-host progression "
                        "containing a process-lineage pivot exists in an F1-suspicious scenario; it can still be ordinary administration. "
                        "cross_host_service_chain_only is informational (ordinary application-service behaviour). no_supporting_path means "
                        "no qualifying path within the available scenario evidence, not proof that movement did not occur. These counts are "
                        "not detection accuracy: no independent lateral-movement ground truth exists."),
}
cross = collections.Counter()
for sid, r in ho.records.items():
    k = ("host_to_host" if r.observed["host_to_host_link_present"] else "no_host_to_host") + ("+iot_host" if r.observed["iot_host_link_present"] else "")
    cross[f"{k} -> {A[sid].assessment}"] += 1
summary["relationship_presence_vs_assessment"] = dict(sorted(cross.items()))


def pick(pred, key=None):
    c = [a for a in A.values() if pred(a)]
    return sorted(c, key=key or (lambda a: (len(ho.records[a.scenario_id].nodes), a.scenario_id)))[0] if c else None


examples = {
    "potential_movement_process_lineage": pick(lambda a: a.assessment == lm.SUPPORTED and a.evidence_types == ["process_lineage"]),
    "potential_movement_both_patterns_36595": A[36595],
    "service_chain_only_single_hop": pick(lambda a: a.assessment == lm.SERVICE_CHAIN_ONLY and max(p["n_hops"] for p in a.supported_paths) == 1),
    "service_chain_only_two_hop_22761": A[22761],
    "no_path_host_to_host_without_pivot": pick(lambda a: a.reason_code == "host_to_host_without_pivot_context" and not a.flags),
    "no_path_destination_side_activity_flagged": pick(lambda a: bool(a.flags)),
    "no_path_single_host": pick(lambda a: a.reason_code == "no_cross_host_communication"),
    "insufficient_two_event_with_host_to_host_link": pick(lambda a: a.reason_code == "fewer_than_3_events" and ho.records[a.scenario_id].observed["host_to_host_link_present"]),
}
ex_out, ex_check = {}, {}
for name, a in examples.items():
    if a is None:
        ex_out[name] = None
        continue
    ts_map = {x["event"]: x["timestamp"] for p in a.paths for x in p["events"]}
    ok = all(H.loc[int(e.split(":")[1]), "timestamp"] == ts for e, ts in ts_map.items() if e.startswith("host:"))
    ex_check[name] = {"scenario_id": a.scenario_id, "assessment": a.assessment, "reason_code": a.reason_code, "flags": a.flags,
                      "evidence_types": a.evidence_types, "events_exist_with_original_timestamps": bool(ok)}
    ex_out[name] = a.to_dict()
summary["example_checks"] = ex_check
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
(OUT / "examples.json").write_text(json.dumps(ex_out, indent=2, default=str))
rows = []
for sid, a in A.items():
    best = (a.qualifying_paths or a.supported_paths or [None])[0]
    rows.append({"scenario_id": sid, "anomaly_score": a.f1["anomaly_score"], "threshold": a.f1["threshold"], "f1_prediction": a.f1["prediction"],
                 "f2_assessment": a.assessment, "reason_code": a.reason_code, "flags": ";".join(a.flags), "evidence_types": ";".join(a.evidence_types),
                 "n_events": len(ho.records[sid].nodes), "qualifying_paths": len(a.qualifying_paths), "service_chain_paths": len(a.service_chain_paths),
                 "best_path_id": best["path_id"] if best else "", "best_path_hops": best["n_hops"] if best else 0,
                 "best_path_hosts": " -> ".join(x["host_id"] for x in best["hosts"]) if best else "",
                 "best_path_events": " ".join(e["event"] for e in best["events"]) if best else ""})
pd.DataFrame(rows).to_csv(OUT / "f2_assessments_all_1499.csv", index=False)
print(json.dumps(summary, indent=2, default=str))
