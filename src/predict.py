import os
import sys
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from dataset import normalize_intensity
from model import build_unet
from utils import load_config, get_device

def predict_volume(model, volume_path: str, device, axis: int = 2, target_size=(256, 256)):
    image = nib.load(volume_path).get_fdata().astype(np.float32)
    original_shape = image.shape
    image_norm = normalize_intensity(image)

    n_slices = original_shape[axis]
    pred_volume = np.zeros_like(image, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for s in range(n_slices):
            sl = np.take(image_norm, s, axis=axis)
            sl_tensor = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).float()

            sl_resized = torch.nn.functional.interpolate(
                sl_tensor, size=target_size, mode="bilinear", align_corners=False
            ).to(device)

            output = model(sl_resized)
            pred = torch.argmax(output, dim=1).squeeze(0)

            orig_hw = sl.shape
            pred_back = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0).float(), size=orig_hw, mode="nearest"
            ).squeeze().long().cpu().numpy()

            if axis == 2:
                pred_volume[:, :, s] = pred_back
            elif axis == 1:
                pred_volume[:, s, :] = pred_back
            else:
                pred_volume[s, :, :] = pred_back
    return pred_volume

def visualize_prediction(image_path: str, mask_path: str, pred_volume: np.ndarray, slice_idx: int, axis: int, save_path: str):
    image = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata() if mask_path else None

    img_sl = np.take(image, slice_idx, axis=axis)
    pred_sl = np.take(pred_volume, slice_idx, axis=axis)

    n_cols = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(img_sl.T, cmap="gray", origin="lower")
    axes[0].set_title("MRI")
    axes[0].axis("off")

    col = 1
    if mask is not None:
        mask_sl = np.take(mask, slice_idx, axis=axis)
        axes[col].imshow(img_sl.T, cmap="gray", origin="lower")
        axes[col].imshow(mask_sl.T, cmap="jet", alpha=0.5, origin="lower")
        axes[col].set_title("Ground Truth")
        axes[col].axis("off")
        col += 1

    axes[col].imshow(img_sl.T, cmap="gray", origin="lower")
    axes[col].imshow(pred_sl.T, cmap="jet", alpha=0.5, origin="lower")
    axes[col].set_title("Prediction")
    axes[col].axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to: {save_path}")


def main(config_path: str, image_path: str, mask_path: str = None, checkpoint_path: str = None, slice_idx: int = None):
    config = load_config(config_path)
    device = get_device()

    if checkpoint_path is None:
        checkpoint_path = f"{config['checkpoints_dir']}/best.pth"

    model = build_unet(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_volume = predict_volume(model, image_path, device, axis=config["axis"], target_size=tuple(config["target_size"]))

    if slice_idx is None:
        slice_idx = pred_volume.shape[config["axis"]] // 2

    case_name = os.path.basename(image_path).replace(".nii.gz", "")
    save_path = f"{config['results_dir']}/figures/{case_name}_slice{slice_idx}.png"
    visualize_prediction(image_path, mask_path, pred_volume, slice_idx, config["axis"], save_path)

    return pred_volume


if __name__ == "__main__":
    main(
        config_path="/content/drive/MyDrive/knee-segmentation-cv/configs/config.yaml",
        image_path="/content/data/raw/imagesTs/oaizib_406_0000.nii.gz", 
        mask_path="/content/data/raw/labelsTs/oaizib_406.nii.gz",
    )