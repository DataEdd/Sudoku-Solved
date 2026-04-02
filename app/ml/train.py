"""
Training script for SudokuCNN digit recognizer.

Usage:
    python -m app.ml.train                    # Full training with defaults
    python -m app.ml.train --epochs 20        # Quick run
    python -m app.ml.train --batch-size 128   # Larger batches
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from app.ml.dataset import create_datasets
from app.ml.model import SudokuCNN, count_parameters

CHECKPOINT_DIR = Path("app/ml/checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "sudoku_cnn.pth"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> np.ndarray:
    model.eval()
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
            matrix[t][p] += 1

    return matrix


def train(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 7,
    device_name: str = "auto",
) -> Dict:
    """Train the SudokuCNN model.

    Returns dict with training history and final metrics.
    """
    # Device selection
    if device_name == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    print(f"Device: {device}")

    # Data
    print("Loading datasets...")
    train_ds, val_ds, test_ds = create_datasets()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Model
    model = SudokuCNN().to(device)
    print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    best_val_acc = 0.0
    no_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train: {train_loss:.4f} / {train_acc:.4f} | "
            f"Val: {val_loss:.4f} / {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s"
        )

        # Early stopping / checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                CHECKPOINT_PATH,
            )
            print(f"  -> Saved checkpoint (val_acc={val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)"
                )
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")

    # Load best model and evaluate on test set
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.4f} (loss: {test_loss:.4f})")

    # Confusion matrix
    cm = compute_confusion_matrix(model, test_loader, device)
    print("\nConfusion Matrix:")
    labels = [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"]
    print("     " + "  ".join(labels))
    for i, row in enumerate(cm):
        print(f"  {i}: " + "  ".join(f"{v:3d}" for v in row))

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / total if total > 0 else 0
        print(f"  {i}: {acc:.4f} ({correct}/{total})")

    # Save results
    results = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "total_time_s": round(total_time, 1),
        "parameters": count_parameters(model),
        "history": history,
        "confusion_matrix": cm.tolist(),
    }
    results_path = CHECKPOINT_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SudokuCNN")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device_name=args.device,
    )
