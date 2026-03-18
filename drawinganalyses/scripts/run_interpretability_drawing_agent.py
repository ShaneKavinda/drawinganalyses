"""
Usage:
- Unlabeled image folder (default):
  python ./drawinganalyses/scripts/run_interpretability_drawing_agent.py --input-dir "C:/path/to/new_images"

- Labeled dataset (optional):
  python ./drawinganalyses/scripts/run_interpretability_drawing_agent.py --input-dir "C:/path/to/new_images" --with-annotations --annotations-file labels.csv
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import Occlusion
from captum.attr import visualization as viz
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from drawinganalyses.config import (
    ANNOTATION_FILE,
    DATASET_NAME,
    INTERPRETABILITY_STORAGE,
    LOCAL_DATA_DIR,
    MODEL_NAME,
    MODELS_STORAGE,
    label_to_str,
)


class FolderDrawingDataset(Dataset):
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, input_dir: Path, transform=None):
        self.input_dir = Path(input_dir)
        self.transform = transform
        self.image_paths = sorted(
            p for p in self.input_dir.rglob("*") if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No supported image files found in: {self.input_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, -1, image_path.name


class CsvDrawingDataset(Dataset):
    def __init__(self, input_dir: Path, annotations_file: str, transform=None):
        self.input_dir = Path(input_dir)
        self.transform = transform
        self.rows = []

        csv_path = self.input_dir / annotations_file
        if not csv_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {csv_path}")

        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                image_name = row[0].strip()
                label_raw = row[1].strip()
                try:
                    label = int(label_raw)
                except ValueError:
                    continue
                if not image_name:
                    continue
                self.rows.append((image_name, label))

        if not self.rows:
            raise ValueError(f"No valid rows found in: {csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image_name, label = self.rows[idx]
        image_path = self.input_dir / image_name
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, image_name


def build_model(num_classes: int, device: torch.device):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(MODELS_STORAGE / MODEL_NAME, map_location=device))
    model.eval()
    return model


def _safe_visualize_attr_multiple(
    attr_array,
    original_array,
    methods,
    signs,
    titles,
    fig_size,
):
    try:
        fig, _ = viz.visualize_image_attr_multiple(
            attr_array,
            original_array,
            methods,
            signs,
            show_colorbar=True,
            outlier_perc=2,
            titles=titles,
            fig_size=fig_size,
            use_pyplot=False,
        )
        return fig
    except AssertionError:
        # Captum can raise when attribution is entirely zero; keep pipeline running.
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(original_array)
        ax.set_title("Original (no non-zero attribution)")
        ax.axis("off")
        return fig


def save_merged_interpretability(
    model,
    inputs,
    pred_label_idx,
    display_inputs,
    output_path: Path,
    true_label: str,
    predicted_label: str,
):
    plt_fig = None
    plt_fig2 = None
    fig3 = None
    try:
        torch.manual_seed(0)
        np.random.seed(0)

        occlusion = Occlusion(model)
        attributions_occ = occlusion.attribute(
            inputs,
            strides=(3, 8, 8),
            target=pred_label_idx,
            sliding_window_shapes=(3, 15, 15),
            baselines=0,
        )

        plt_fig = _safe_visualize_attr_multiple(
            np.transpose(attributions_occ.squeeze().detach().cpu().numpy(), (1, 2, 0)),
            np.transpose(display_inputs.numpy(), (1, 2, 0)),
            ["original_image", "heat_map", "heat_map", "masked_image"],
            ["all", "positive", "negative", "positive"],
            ["Original", "Positive Attribution", "Negative Attribution", "Masked"],
            (18, 6),
        )

        occlusion = Occlusion(model)
        attributions_occ1 = occlusion.attribute(
            inputs,
            strides=(3, 50, 50),
            target=pred_label_idx,
            sliding_window_shapes=(3, 60, 60),
            baselines=0,
        )

        plt_fig2 = _safe_visualize_attr_multiple(
            np.transpose(attributions_occ1.squeeze().detach().cpu().numpy(), (1, 2, 0)),
            np.transpose(display_inputs.numpy(), (1, 2, 0)),
            ["original_image", "heat_map", "masked_image"],
            ["all", "positive", "positive"],
            ["Original", "Positive Attribution", "Masked"],
            (18, 6),
        )

        canvas1 = FigureCanvasAgg(plt_fig)
        canvas2 = FigureCanvasAgg(plt_fig2)
        canvas1.draw()
        canvas2.draw()
        fig1_rgba = np.asarray(canvas1.buffer_rgba())
        fig2_rgba = np.asarray(canvas2.buffer_rgba())

        fig3 = plt.figure(figsize=(36, 12))
        ax3 = fig3.add_subplot(2, 1, 1)
        ax4 = fig3.add_subplot(2, 1, 2)

        ax3.imshow(fig1_rgba)
        ax4.imshow(fig2_rgba)
        ax3.set_title(f"True label : {true_label}, Predicted label : {predicted_label}")
        ax3.axis("off")
        ax4.axis("off")

        fig3.savefig(output_path)
    finally:
        if plt_fig is not None:
            plt.close(plt_fig)
        if plt_fig2 is not None:
            plt.close(plt_fig2)
        if fig3 is not None:
            plt.close(fig3)


def run_inference_and_interpretability(
    dataloader,
    model,
    class_names,
    output_dir: Path,
    device: torch.device,
    max_images: Optional[int] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_dir / "predictions.csv"

    with open(predictions_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "predicted_label", "prediction_score", "true_label", "is_correct"])

        for idx, (inputs, labels, names) in enumerate(tqdm(dataloader), start=1):
            if max_images is not None and idx > max_images:
                break

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output = model(inputs)
            output = F.softmax(output, dim=1)
            prediction_score, pred_label_idx = torch.topk(output, 1)
            pred_label_idx.squeeze_()

            pred_index = pred_label_idx.item()
            predicted_label = class_names[pred_index]
            score = prediction_score.squeeze().item()

            label_idx = labels.item()
            has_ground_truth = label_idx >= 0
            true_label = label_to_str.get(label_idx, "") if has_ground_truth else ""
            is_correct = (predicted_label == true_label) if has_ground_truth else ""

            print(f"Image: {names[0]}")
            print(f"Predicted: {predicted_label} ({score:.6f})")
            if has_ground_truth:
                print(f"Ground truth: {true_label}")

            display_inputs = (inputs.squeeze().detach().cpu() * 0.5 + 0.5).clamp(0, 1)
            output_path = output_dir / f"{Path(names[0]).stem}_{idx - 1}.png"
            save_merged_interpretability(
                model=model,
                inputs=inputs,
                pred_label_idx=pred_label_idx,
                display_inputs=display_inputs,
                output_path=output_path,
                true_label=true_label if has_ground_truth else "N/A",
                predicted_label=predicted_label,
            )

            writer.writerow([names[0], predicted_label, score, true_label, is_correct])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference + interpretability on a new image set."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=LOCAL_DATA_DIR / DATASET_NAME,
        help="Folder containing images.",
    )
    parser.add_argument(
        "--annotations-file",
        default=ANNOTATION_FILE,
        help="CSV file name in input-dir with rows formatted as: image_name,label",
    )
    parser.add_argument(
        "--with-annotations",
        action="store_true",
        help="Use labels CSV for evaluation metadata. Default behavior is unlabeled classification.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(INTERPRETABILITY_STORAGE) / "drawing_agent",
        help="Where interpretability images and predictions.csv are saved.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((256, 256)),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, running on CPU.")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if args.with_annotations:
        if not (input_dir / args.annotations_file).exists():
            raise FileNotFoundError(
                f"--with-annotations was set but CSV was not found: {input_dir / args.annotations_file}"
            )
        dataset = CsvDrawingDataset(
            input_dir=input_dir,
            annotations_file=args.annotations_file,
            transform=transform,
        )
        print(f"Loaded labeled dataset from: {input_dir / args.annotations_file}")
    else:
        dataset = FolderDrawingDataset(input_dir=input_dir, transform=transform)
        print(f"Loaded unlabeled image folder: {input_dir}")

    num_workers = min(8, os.cpu_count() or 1)
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(dataset, **loader_kwargs)

    class_names = [label_to_str[k] for k in sorted(label_to_str.keys())]
    model = build_model(num_classes=len(class_names), device=device)

    run_inference_and_interpretability(
        dataloader=dataloader,
        model=model,
        class_names=class_names,
        output_dir=Path(args.output_dir),
        device=device,
        max_images=args.max_images,
    )

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
