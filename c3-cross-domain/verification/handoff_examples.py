"""Build the real F1 -> F2 handoff from the actual pipeline results and save summary + example records
to verification/f1_f2/ (no research artifact folder is written)."""
import json, pickle, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xdr_c3 import graph_builder as gb, inference as inf, f1_f2_handoff as h, scenario_grouping as sg, stage2_validate as s2

t = time.time()
data = s2.load_and_validate(); s3b = sg.run_stage3b(data)
gc = gb.build_graphs_from_stage3b(data, s3b); det = inf.Stage5Detector.load()
res = det.predict_collection(gc); ho = h.build_f1_f2_handoff(data, s3b, res, det.contract)
recs = ho.records.values()
facts = {"summary": ho.summary(), "all_records_f2_ready": all(r.f2_ready for r in recs),
         "records_with_iot_host_match_edge": sum(r.observed["iot_host_link_present"] for r in recs),
         "records_with_connect_accept_edge": sum(r.observed["host_to_host_link_present"] for r in recs),
         "records_with_both": sum(r.observed["iot_host_link_present"] and r.observed["host_to_host_link_present"] for r in recs),
         "events_in_suspicious_scenarios": sum(len(r.nodes) for r in recs), "edges_in_suspicious_scenarios": sum(len(r.edges) for r in recs),
         "evidence_links_in_suspicious_scenarios": sum(len(r.evidence) for r in recs), "seconds_total": round(time.time() - t, 1)}
both = [r for r in recs if r.observed["iot_host_link_present"] and r.observed["host_to_host_link_present"]]
small = sorted(both, key=lambda r: len(r.nodes))[:1]
top = sorted(recs, key=lambda r: -r.f1["anomaly_score"])[:1]
(ROOT / "verification" / "f1_f2" / "summary.json").write_text(json.dumps(facts, indent=2))
(ROOT / "verification" / "f1_f2" / "example_records.json").write_text(json.dumps([r.to_dict() for r in small + top], indent=2))
print(json.dumps(facts, indent=2))
for r in small + top:
    print("\n=== scenario", r.scenario_id, r.f1); print(r.nodes[["local_index","node","timestamp","host_ip","model_node_error"]].round(4).to_string())
    print(r.edges[["edge_index","node_a","node_b","relation","same_host","short_gap","min_gap_sec","model_edge_error_mean"]].round(4).to_string())
    print(r.evidence.to_string()); print(r.observed)
