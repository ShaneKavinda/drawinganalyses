import os
import warnings

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
from torch.utils.data import DataLoader, random_split
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
from drawinganalyses.datasets.drawings_pytorch import DrawingDataset

warnings.filterwarnings("ignore")
plt.ioff()


def interpretability_save_gpu(dataloader, model, label_to_str, class_names, split, device):
    """
    Apply Captum occlusion and save interpreted images on disk using GPU execution.
    """
    total_count = 0
    output_dir = LOCAL_DATA_DIR / INTERPRETABILITY_STORAGE / split
    output_dir.mkdir(parents=True, exist_ok=True)

    for inputs, labels, _ in tqdm(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        true_label = label_to_str[labels.item()]
        output = model(inputs)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        pred_label_idx.squeeze_()
        predicted_label = class_names[pred_label_idx.item()]

        print("Predicted:", predicted_label, "(", prediction_score.squeeze().item(), ")")
        print("Ground truth:", true_label)
        print("total count :", total_count)

        plt_fig = None
        plt_fig2 = None
        fig3 = None
        try:
            torch.manual_seed(0)
            np.random.seed(0)

            occlusion = Occlusion(model)
            copy_inputs = inputs
            display_inputs = (inputs.squeeze().detach().cpu() * 0.5 + 0.5).clamp(0, 1)

            attributions_occ = occlusion.attribute(
                inputs,
                strides=(3, 8, 8),
                target=pred_label_idx,
                sliding_window_shapes=(3, 15, 15),
                baselines=0,
            )

            plt_fig, _ = viz.visualize_image_attr_multiple(
                np.transpose(attributions_occ.squeeze().detach().cpu().numpy(), (1, 2, 0)),
                np.transpose(display_inputs.numpy(), (1, 2, 0)),
                ["original_image", "heat_map", "heat_map", "masked_image"],
                ["all", "positive", "negative", "positive"],
                show_colorbar=True,
                outlier_perc=2,
                titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                fig_size=(18, 6),
                use_pyplot=False,
            )

            occlusion = Occlusion(model)
            attributions_occ1 = occlusion.attribute(
                copy_inputs,
                strides=(3, 50, 50),
                target=pred_label_idx,
                sliding_window_shapes=(3, 60, 60),
                baselines=0,
            )

            plt_fig2, _ = viz.visualize_image_attr_multiple(
                np.transpose(attributions_occ1.squeeze().detach().cpu().numpy(), (1, 2, 0)),
                np.transpose(display_inputs.numpy(), (1, 2, 0)),
                ["original_image", "heat_map", "masked_image"],
                ["all", "positive", "positive"],
                show_colorbar=True,
                outlier_perc=2,
                titles=["Original", "Positive Attribution", "Masked"],
                fig_size=(18, 6),
                use_pyplot=False,
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
            ax3.set_title(
                "True label : {true_label}, Predicted label : {predicted_label}".format(
                    true_label=true_label, predicted_label=predicted_label
                )
            )
            ax3.axis("off")
            ax4.axis("off")

            fig3.savefig(output_dir / f"{split}_{total_count}.png")
            total_count += 1
            print("saved figure ", total_count)

        except AssertionError:
            print("continue")
            continue
        finally:
            if plt_fig is not None:
                plt.close(plt_fig)
            if plt_fig2 is not None:
                plt.close(plt_fig2)
            if fig3 is not None:
                plt.close(fig3)


def main():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((256, 256)),
        ]
    )

    generator1 = torch.Generator().manual_seed(42)
    dataset = DrawingDataset(
        dataset_name=DATASET_NAME,
        annotations_file=ANNOTATION_FILE,
        data_dir=LOCAL_DATA_DIR,
        label_to_str=label_to_str,
        transform=transform,
    )
    trainset, valset, testset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator1)

    class_names = list(label_to_str.values())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    if use_cuda:
        cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, running on CPU.")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)
    model.load_state_dict(torch.load(MODELS_STORAGE / MODEL_NAME, map_location=device))
    model.eval()

    num_workers = min(8, os.cpu_count() or 1)
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    trainloader = DataLoader(trainset, **loader_kwargs)
    valloader = DataLoader(valset, **loader_kwargs)
    testloader = DataLoader(testset, **loader_kwargs)

    print("Applying to the Training set")
    interpretability_save_gpu(trainloader, model, label_to_str, class_names, "train", device)
    print("Applying to the Validation set")
    interpretability_save_gpu(valloader, model, label_to_str, class_names, "valid", device)
    print("Applying to the Test set")
    interpretability_save_gpu(testloader, model, label_to_str, class_names, "test", device)


if __name__ == "__main__":
    main()
