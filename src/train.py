import os
import sys
import json
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(__file__))
from dataset import gather_processed_pairs, build_slice_index, KneeDataset, ChunkedVolumeSampler
from model import build_unet
from utils import (
    set_seed, load_config, get_device,
    save_checkpoint, save_best_model, save_results, load_checkpoint, EarlyStopping,
)
from transforms import Compose, RandomFlip, RandomRotate90, RandomIntensityShift
from utils import plot_training_progress


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch=None,
                     checkpoint_path=None, checkpoint_every_n_steps=None, last_metric=0.0):
    model.train()
    total_loss = 0

    for step, (images, masks) in enumerate(tqdm(loader, desc="Train", leave=False)):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        if checkpoint_every_n_steps and (step + 1) % checkpoint_every_n_steps == 0:
            save_checkpoint(model, optimizer, epoch, last_metric, path=checkpoint_path)

    return total_loss / len(loader.dataset)


def validate(model, loader, loss_fn, dice_metric, device, num_classes):
    model.eval()
    total_loss = 0
    dice_metric.reset()

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1, keepdim = True)
            preds_onehot = one_hot(preds, num_classes = num_classes)
            masks_onehot = one_hot(masks, num_classes = num_classes)
            dice_metric(y_pred=preds_onehot, y=masks_onehot)

    val_loss = total_loss / len(loader.dataset)
    val_dice_per_class = dice_metric.aggregate(reduction="mean_batch")
    val_dice_mean = val_dice_per_class.mean().item()

    return val_loss, val_dice_per_class.tolist(), val_dice_mean


def main(config_path: str, resume: bool = False):
    config = load_config(config_path)
    set_seed(config["seed"])
    device = get_device()
    print(f"Using device: {device}")

    pairs = gather_processed_pairs(config["processed_dir"])

    train_pairs, val_pairs = train_test_split(
        pairs, test_size=config["val_split"], random_state=config["seed"]
    )

    train_slice_idx = build_slice_index(train_pairs, axis=config["axis"])
    val_slice_idx = build_slice_index(val_pairs, axis=config["axis"])

    train_transform = Compose([
        RandomFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomIntensityShift(p=0.5, shift_range=0.1),
    ])

    train_ds = KneeDataset(
        train_pairs, train_slice_idx, axis=config["axis"],
        target_size=config["target_size"], transform=train_transform,
    )
    val_ds = KneeDataset(
        val_pairs, val_slice_idx, axis=config["axis"],
        target_size=config["target_size"],
    )

    train_sampler = ChunkedVolumeSampler(
        train_slice_idx, chunk_size=config.get("shuffle_chunk_volumes", 32), seed=config["seed"]
    )
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], sampler=train_sampler,
        num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=2, persistent_workers=True, pin_memory=True, prefetch_factor=4
    )

    model = build_unet(config).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=config["scheduler_patience"], factor=0.5)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        mode=config["early_stopping_mode"]
    )

    checkpoints_dir = config["checkpoints_dir"]
    results_dir = config["results_dir"]
    history = []
    last_val_dice_mean = 0.0
    start_epoch = 0

    if resume:
        loaded_epoch, loaded_metric = load_checkpoint(
            model, optimizer, f"{checkpoints_dir}/last.pth", device
        )
        start_epoch = loaded_epoch + 1
        last_val_dice_mean = loaded_metric

        history_path = f"{results_dir}/metrics/train_history.json"
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)["history"]
            past_scores = [h["val_dice_mean"] for h in history]
            if past_scores:
                early_stopping.best_score = (
                    max(past_scores) if config["early_stopping_mode"] == "max" else min(past_scores)
                )

        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, config["epochs"]):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch=epoch,
            checkpoint_path=f"{checkpoints_dir}/last.pth",
            checkpoint_every_n_steps=config.get("checkpoint_every_n_steps"),
            last_metric=last_val_dice_mean,
        )
        val_loss, val_dice, val_dice_mean = validate(
            model, val_loader, loss_fn, dice_metric, device, config["out_channels"]
        )
        last_val_dice_mean = val_dice_mean
        scheduler.step(val_dice_mean)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice_mean:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice_mean": val_dice_mean,
            "val_dice_per_class": val_dice,
        })

        save_checkpoint(model, optimizer, epoch, val_dice_mean, path=f"{checkpoints_dir}/last.pth")

        is_best = early_stopping.step(val_dice_mean)
        if is_best:
            save_best_model(model, epoch, val_dice_mean, path=f"{checkpoints_dir}/best.pth")

        save_results({"history": history}, path=f"{results_dir}/metrics/train_history.json")
        plot_training_progress(history, save_path=f"{results_dir}/figures/training_progress.png")

        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break


if __name__ == "__main__":
    main(config_path="/content/drive/MyDrive/knee-segmentation-cv/configs/config.yaml")
        