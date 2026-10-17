"""Project-wide paths and constants (local paths only; no Colab / Drive locations).

Values here mirror the research notebooks and must not be changed to alter methodology.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw telemetry (Component 1 = host, Component 2 = IoT). Override the folder with XDR_C3_DATA_DIR if needed.
DATA_INPUT_DIR = Path(os.environ.get("XDR_C3_DATA_DIR", PROJECT_ROOT / "Data" / "input"))
HOST_CSV = DATA_INPUT_DIR / "Host.csv"
IOT_CSV = DATA_INPUT_DIR / "IOT_Network_Flow.csv"

# Stage 2 / Stage 3 notebooks: strict day-first timestamp format.
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"

# --------------------------------------------------------------------------- saved research artifacts (read-only)
STAGES_DIR = PROJECT_ROOT / "Stages"
STAGE4_OUT = STAGES_DIR / "Stage 4" / "Outputs"
FEATURE_NAMES_CSV = STAGE4_OUT / "Graph Data" / "feature_names_v6v2 (1).csv"
SCALER_CSV = STAGE4_OUT / "Splits" / "scaler_b1_v6v2.csv"
SPLIT_NPZ = {
    "train": STAGE4_OUT / "Splits" / "graphs_b1_train_v6v2 (4).npz",
    "validation": STAGE4_OUT / "Splits" / "graphs_b1_validation_v6v2 (4).npz",
    "test_normal": STAGE4_OUT / "Splits" / "graphs_b1_test_normal_v6v2 (4).npz",
    "test_attack": STAGE4_OUT / "Splits" / "graphs_b1_test_attack_v6v2 (4).npz",
}
STAGE5_DIR = STAGES_DIR / "Stage 5_R-GCN"
MODEL_CONFIG_JSON = STAGE5_DIR / "training" / "configuration.json"
CHECKPOINT_PATH = STAGE5_DIR / "training" / "Checkpoints" / "epoch_040.pt"      # selected epoch (locked_selection.json)
LOCKED_SELECTION_JSON = STAGE5_DIR / "validation" / "validation" / "locked_selection.json"

# --------------------------------------------------------------------------- Stage 5 inference
CHECKPOINT_SHA256 = "efc13fee5e56e717a66ec1f89e3077528ba00f2f3097d5d35bc22647178c53d2"   # trusted local epoch_040.pt
VALIDATION_SCORES_CSV = STAGE5_DIR / "validation" / "validation" / "per_graph_validation_scores.csv"
CACHE_DIR = PROJECT_ROOT / "cache" / "stage5"        # application cache; never inside the research artifact folders
