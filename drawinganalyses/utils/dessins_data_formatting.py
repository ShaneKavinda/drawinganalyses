import os
from pathlib import Path

import pandas as pd

from drawinganalyses.config import LOCAL_DATA_DIR


data_dir = LOCAL_DATA_DIR / "Dessins pour ML"
labels = {
    "Crèche (2-3yo)": 0,
    "Petite section (3-4yo)": 1,
    "Moyenne section (4-5yo)": 2,
    "Grande section (5-6yo)": 3,
    "CE1 (7-8yo)": 4,
    "CE2 (8-9yo)": 5,
    "CM1 (9-10yo)": 6,
    "CM2 (10-11yo)": 7,
    "Adults NOVICES": 8,
    "Adults EXPERT": 9,
}

list_tuple_annotations = []
for root, _, files in os.walk(data_dir):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext in {".jpg", ".jpeg", ".png"}:
            rel_path = (Path(root) / file).relative_to(data_dir).as_posix()
            label_folder = rel_path.split("/")[0]
            if label_folder in labels:
                list_tuple_annotations.append((rel_path, labels[label_folder]))

list_tuple_annotations.sort(key=lambda item: item[0])
df = pd.DataFrame(list_tuple_annotations, columns=["name", "label"])
df.to_csv(data_dir / "labels.csv", index=False)

