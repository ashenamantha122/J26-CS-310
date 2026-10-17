"""Compare build_links_from_stage2 on the real CSVs with the saved Stage 3 reference. Writes verification/stage3/*.json."""
import json, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xdr_c3 import relationship_mapping as rm, stage2_validate as s2

REF = ROOT / "Stages" / "Stage 3" / "Outputs" / "links_all_v6v2.csv"
t0 = time.time()
data = s2.load_and_validate()
links = rm.build_links_from_stage2(data)
secs = time.time() - t0
ref = pd.read_csv(REF)
res = {"generated_rows": len(links), "reference_rows": len(ref), "seconds": round(secs, 1),
       "columns_equal": list(links.columns) == list(ref.columns),
       "type_counts_generated": links.link_type.value_counts().sort_index().to_dict(),
       "type_counts_reference": ref.link_type.value_counts().sort_index().to_dict()}
res["type_counts_equal"] = res["type_counts_generated"] == res["type_counts_reference"]
# CSV round trip with the notebook's own writer settings, then exact comparison in original order
tmp = ROOT / "verification" / "stage3" / "links_all_regenerated.csv"
links.to_csv(tmp, index=False, lineterminator="\n")
back = pd.read_csv(tmp)
res["row_by_row_equal_original_order"] = bool(back.equals(ref))
res["bytes_identical_to_reference"] = tmp.read_bytes() == REF.read_bytes()
key = ["node_a", "node_b", "link_type"]
res["key_sets_equal"] = set(map(tuple, links[key].to_numpy())) == set(map(tuple, ref[key].to_numpy()))
m = links.merge(ref, on=key, suffixes=("_g", "_r"))
res["merged_rows"] = len(m)
res["points_mismatch"] = int((m.points_g != m.points_r).sum())
res["max_abs_gap_diff"] = float((m.gap_sec_g - m.gap_sec_r).abs().max())
res["unsorted_equal_after_sort"] = bool(links.sort_values(key).reset_index(drop=True).equals(
    ref.sort_values(key).reset_index(drop=True)))
(ROOT / "verification" / "stage3" / "comparison.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
