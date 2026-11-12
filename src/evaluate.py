import os
import sys
import warnings
import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(__file__))
from dataset import gather_pairs, build_slice_index, KneeDataset
from model import build_unet
from utils import load_config, get_device, save_results

CLASS_NAMES = ["background", "femur", "femoral_cartilage", "tibia", "medial_tibial_cartilage", "lateral_tibial_cartilage"]

def evaluate(config_path: str, checkpoint_path: str = None, results_name: str = "test_evaluation"):
    config = load_config(config_path)
    device = get_device()

    if checkpoint_path is None:
        checkpoint_path = f"{config['checkpoints_dir']}/best.pth"

    images_dir = os.path.join(config["data_dir"], "imagesTs")
    labels_dir = os.path.join(config["data_dir"], "labelsTs")
    pairs = gather_pairs(images_dir, labels_dir)
    test_slice_index = build_slice_index(pairs, axis=config["axis"])
    test_ds = KneeDataset(
        pairs, test_slice_index, axis=config["axis"], target_size=config["target_size"]
    )
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    model = build_unet(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {checkpoint_path} (epoch={checkpoint['epoch']}, val_dice={checkpoint['metric']:.4f})")

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd_metric = HausdorffDistanceMetric(include_background=False, reduction="mean_batch", percentile=95)

    num_classes = config["out_channels"]

    with warnings.catch_warnings():
        # empty-class slices are expected here (thin structures like cartilage aren't in
        # every slice) and just spam UserWarnings from monai's Hausdorff/edge computation
        warnings.simplefilter("ignore")

        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Evaluate"):
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1, keepdim=True)

                preds_onehot = one_hot(preds, num_classes=num_classes)
                masks_onehot = one_hot(masks, num_classes=num_classes)

                dice_metric(y_pred=preds_onehot, y=masks_onehot)
                hd_metric(y_pred=preds_onehot.cpu(), y=masks_onehot.cpu())

    dice_per_class = dice_metric.aggregate(reduction='mean_batch').tolist()
    hd_per_class = hd_metric.aggregate(reduction="mean_batch").tolist()

    results = {
        "checkpoint": checkpoint_path,
        "dice_per_class": dict(zip(CLASS_NAMES[1:], dice_per_class)),
        "dice_mean": sum(dice_per_class) / len(dice_per_class),
        "hausdorff95_per_class": dict(zip(CLASS_NAMES[1:], hd_per_class)),
    }

    print("\n=== Results on test ===")
    for name, val in results["dice_per_class"].items():
        print(f"  Dice [{name}]: {val:.4f}")
    print(f"  Mean Dice: {results['dice_mean']:.4f}")
    for name, val in results["hausdorff95_per_class"].items():
        print(f" Hausdorff95 [{name}]: {val:.4f}")

    save_results(results, path=f"{config['results_dir']}/metrics/{results_name}.json")
    return results

if __name__ == "__main__":
    evaluate(config_path="/content/drive/MyDrive/knee-segmentation-cv/configs/config.yaml")