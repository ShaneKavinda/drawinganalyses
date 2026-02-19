import os
from pathlib import Path


# Default to repo root when MADE_DATA_DIR is not set.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = Path(os.environ.get("MADE_DATA_DIR", str(DEFAULT_DATA_DIR)))
assert LOCAL_DATA_DIR.exists(), f"Data directory not found: {LOCAL_DATA_DIR}"

MODELS_STORAGE = LOCAL_DATA_DIR / "models"
MODELS_STORAGE.mkdir(parents=True, exist_ok=True)

# Local dataset provided in this workspace.
DATASET_NAME = "Dessins pour ML"
ANNOTATION_FILE = "labels.csv"
MODEL_NAME = "Dessins_pour_ML_model.pth"
label_to_str = {
    0: "Creche_2_3yo",
    1: "Petite_section_3_4yo",
    2: "Moyenne_section_4_5yo",
    3: "Grande_section_5_6yo",
    4: "CE1_7_8yo",
    5: "CE2_8_9yo",
    6: "CM1_9_10yo",
    7: "CM2_10_11yo",
    8: "Adults_NOVICES",
    9: "Adults_EXPERT",
}

INTERPRETABILITY_STORAGE = LOCAL_DATA_DIR / f"{DATASET_NAME}_interpretability"
INTERPRETABILITY_STORAGE.mkdir(parents=True, exist_ok=True)
