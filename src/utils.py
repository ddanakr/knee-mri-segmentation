import os 
import random
import yaml
import json
import numpy as np
import torch
import matplotlib.pyplot as plt



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, optimizer, epoch: int, metric: float, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": metric
    }, path)

def save_best_model(model, epoch: int, metric: float, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metric": metric,
    }, path)
    print(f"[best model] epoch={epoch}, metric={metric:.4f} -> saved to {path}")

def load_checkpoint(model, optimizer,  path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded: epoch={checkpoint['epoch']}, metric={checkpoint['metric']:.4f}")
    return checkpoint["epoch"], checkpoint["metric"]

def save_results(results: dict, path:str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


class EarlyStopping:

    def __init__(self, patience: int=5, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_score: float) -> bool:
        is_best = False
        if self.best_score is None:
            self.best_score = current_score
            is_best = True
        elif self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
            is_best = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return is_best

    def _is_improvement(self, current_score: float) -> bool:
        if self.mode == "max":
            return current_score > self.best_score + self.min_delta
        else:
            return current_score < self.best_score - self.min_delta

def plot_training_progress(history: list, save_path: str):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_dice = [h["val_dice_mean"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, val_dice, label="val_dice", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Validation Dice")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close(fig)