"""
Architecture ablation for the SudokuCNN digit recognizer.

Trains every configuration in a 3 × 3 × 3 grid under an identical protocol
(same seed, same data, same optimizer, same LR schedule, same epoch budget,
same early-stopping patience) and evaluates each on:

    1. Synthetic test split (MNIST test + held-out empty cells)
    2. Real-photo filled/empty accuracy on the 38-image ground-truth set
       (using ground-truth corners to isolate OCR from detection quality).

Results are streamed to ``evaluation/ablation_results.json`` after every
single config so that partial progress survives a crash. Re-running with
an existing results file resumes where it left off.

Grid axes:
    depth    ∈ {2, 3, 4}                                  (conv block count)
    channels ∈ {small, medium, large}                     (width per block)
    dropout  ∈ {0.2, 0.3, 0.5}                            (classifier head)

Channel schemes:
    small  : [16, 32 ...]            (lower-capacity variant)
    medium : [32, 64, 128 ...]       (production baseline width at depth=3)
    large  : [64, 128, 256 ...]      (higher-capacity variant)

Training protocol (identical across all 27 configs, mirrors v5.1 production):
    epochs = 20             (reduced from production's 30 for budget)
    batch_size = 64         (matches production)
    optimizer = Adam(lr=1e-3)
    scheduler = CosineAnnealingLR(T_max=epochs)
    patience = 7            (matches production early-stopping window)
    seed = 42               (reset before every config → identical data + init)
    loss = CrossEntropyLoss(weight=[2.0, 1.0, ..., 1.0])
                            (class-weighted to match v5.1 — compensates
                             for class 0's ~7% share after MNIST 0s were
                             dropped in v4.2; without this the ablation
                             trains under a different loss than production
                             and the baseline row would not match the real
                             v5.1 numbers)

Evaluation (three metrics per config):
    synthetic_test_acc_clean : clean synthetic test split (reproducible,
                               comparable to historical checkpoints)
    synthetic_test_acc_aug   : newsprint-augmented synthetic test split
                               (L1 realistic proxy from the 2026-04-11
                               pipeline review — better correlates with
                               real-photo performance than the clean split)
    real_filled / real_empty : 38-image GT set, using GT corners (isolates
                               classifier quality from detection quality)
    Real-photo inference uses confidence_threshold=0.50 to match
    CNNRecognizer's v5.1 default.

Production baseline equivalence:
    The production SudokuCNN is exactly
        depth=3, channels=[32,64,128], dropout=0.3, hidden_fc=64
    which corresponds to ablation config "d3_c-medium_drop0.3". That config
    is the cross-validation anchor for the ablation: its numbers should
    closely track ``app/ml/checkpoints/training_results.json`` (modulo the
    epoch-budget reduction from 30 → 20).

Usage:
    python -m evaluation.ablation                   # Full 3×3×3 grid (27 configs)
    python -m evaluation.ablation --only-baseline   # Smoke test: baseline only
    python -m evaluation.ablation --max-configs 5   # Run only first 5 (debugging)
    python -m evaluation.ablation --dropout-only 0.3 # 9-config diagonal at fixed dropout

Output:
    evaluation/ablation_results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.extraction import extract_cells, perspective_transform  # noqa: E402
from app.ml.dataset import AugmentedDataset, create_datasets  # noqa: E402
from app.ml.model_variants import SudokuCNNAblation, count_parameters  # noqa: E402
from app.ml.train import evaluate, train_one_epoch  # noqa: E402

# ---- Grid configuration ----------------------------------------------------

DEPTHS = [2, 3, 4]
CHANNEL_SCHEMES: Dict[str, Dict[int, List[int]]] = {
    "small":  {2: [16, 32],       3: [16, 32, 64],    4: [16, 32, 64, 128]},
    "medium": {2: [32, 64],       3: [32, 64, 128],   4: [32, 64, 128, 256]},
    "large":  {2: [64, 128],      3: [64, 128, 256],  4: [64, 128, 256, 512]},
}
DROPOUTS = [0.2, 0.3, 0.5]
CHANNEL_NAMES = ["small", "medium", "large"]

# ---- Training protocol -----------------------------------------------------

EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 7                 # Matches v5.1 production (was 5 in pre-2026-04-11 version)
SEED = 42
HIDDEN_FC = 64

# v5.1 production loss: class 0 (empty) weight is 2× because it's only
# ~7% of the v4.2 training pool after MNIST 0s were dropped. Boosting
# the weight to 2.0 restores class 0's effective gradient share to
# ~14%, matching the v3 baseline. See app/ml/train.py lines 164-178.
CLASS_WEIGHTS = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# ---- Real-photo eval protocol ---------------------------------------------

CONFIDENCE_THRESHOLD = 0.50  # Mirror CNNRecognizer default (v5.1, 2026-04-11)
EMPTY_THRESHOLD = 0.03       # Mirror CNNRecognizer._is_empty (mean < thresh*255)

# ---- IO --------------------------------------------------------------------

OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "ablation_results.json"
GT_PATH = PROJECT_ROOT / "evaluation" / "ground_truth.json"
IMAGES_DIR = PROJECT_ROOT / "Examples" / "Ground Example"


# ---- Helpers ---------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Deterministic seeding across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: MPS does not expose a separate seed API; torch.manual_seed()
    # is sufficient for reproducibility on Apple Silicon.


def preprocess_cell(cell: np.ndarray) -> np.ndarray:
    """Preprocess a cell crop into a 28×28 grayscale array.

    Mirrors ``app/ml/recognizer.py::CNNRecognizer._preprocess`` exactly
    so that real-photo accuracy is measured under the same conditions
    the production pipeline uses.
    """
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    h, w = cell.shape
    margin_y = max(1, h // 10)
    margin_x = max(1, w // 10)
    cell = cell[margin_y:-margin_y, margin_x:-margin_x]

    cell = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
    cell = 255 - cell  # invert (white-on-black like MNIST)
    cell = cv2.normalize(cell, None, 0, 255, cv2.NORM_MINMAX)
    return cell


def load_gt_cells() -> List[Dict[str, Any]]:
    """Preprocess every ground-truth image's cells once, up front.

    Returns a list of dicts, one per image, each containing the
    81 preprocessed ``(28, 28)`` cell arrays and the per-cell GT labels.
    Using GT corners instead of ``detect_grid`` isolates OCR quality
    from detection quality — the goal is to measure the classifier
    in isolation.
    """
    with open(GT_PATH) as f:
        entries = json.load(f)["images"]

    loaded: List[Dict[str, Any]] = []
    for entry in entries:
        fname = Path(entry["path"]).name
        img_path = IMAGES_DIR / fname
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # GT corners: use [0, 3, 15, 12] from the 16-point annotation
        c16 = entry["corners_16"]
        corners = np.array(
            [c16[0], c16[3], c16[15], c16[12]], dtype=np.float32
        )
        contour = corners.reshape(4, 1, 2).astype(np.float32)
        warped = perspective_transform(img, contour)
        cells = extract_cells(warped)

        processed = np.stack([preprocess_cell(c) for c in cells])  # (81, 28, 28)
        loaded.append(
            {
                "filename": fname,
                "processed": processed,
                "gt_grid": entry["grid"],
            }
        )

    return loaded


def evaluate_real_photos(
    model: nn.Module,
    device: torch.device,
    gt_cells: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the model over every preprocessed GT cell and score."""
    model.eval()
    total_filled = 0
    correct_filled = 0
    total_empty = 0
    correct_empty = 0
    hallucinated = 0
    missed_filled = 0
    wrong_digit = 0

    with torch.no_grad():
        for image in gt_cells:
            processed = image["processed"]  # (81, 28, 28)
            gt_grid = image["gt_grid"]

            # Batch through the model
            batch = (processed.astype(np.float32) / 255.0)[:, np.newaxis, :, :]
            tensor = torch.from_numpy(batch).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for idx in range(81):
                row = idx // 9
                col = idx % 9
                gt_val = gt_grid[row][col]
                gt_digits = gt_val if isinstance(gt_val, list) else [gt_val]
                gt_filled = any(v != 0 for v in gt_digits)

                # Empty filter (matches CNNRecognizer._is_empty)
                cell_mean = float(processed[idx].mean())
                is_empty_filtered = cell_mean < EMPTY_THRESHOLD * 255

                if is_empty_filtered:
                    pred = 0
                else:
                    digit_probs = probs[idx][1:]  # classes 1-9 only
                    best = int(digit_probs.argmax()) + 1
                    conf = float(digit_probs.max())
                    pred = best if conf >= CONFIDENCE_THRESHOLD else 0

                if gt_filled:
                    total_filled += 1
                    if pred in gt_digits:
                        correct_filled += 1
                    elif pred == 0:
                        missed_filled += 1
                    else:
                        wrong_digit += 1
                else:
                    total_empty += 1
                    if pred == 0:
                        correct_empty += 1
                    else:
                        hallucinated += 1

    filled_acc = correct_filled / total_filled if total_filled else 0.0
    empty_acc = correct_empty / total_empty if total_empty else 0.0
    return {
        "filled_accuracy": round(filled_acc, 4),
        "empty_accuracy": round(empty_acc, 4),
        "total_filled": total_filled,
        "correct_filled": correct_filled,
        "total_empty": total_empty,
        "correct_empty": correct_empty,
        "missed_filled": missed_filled,
        "wrong_digit": wrong_digit,
        "hallucinated": hallucinated,
        "images_evaluated": len(gt_cells),
    }


def train_one_config(
    config: Dict[str, Any],
    device: torch.device,
    gt_cells: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Train and evaluate a single configuration. Returns metrics dict."""
    set_seed(SEED)

    model = SudokuCNNAblation(
        depth=config["depth"],
        channels=config["channels"],
        dropout=config["dropout"],
        hidden_fc=HIDDEN_FC,
    ).to(device)
    params = count_parameters(model)

    train_ds, val_ds, test_ds = create_datasets(seed=SEED)
    # L1 realistic-split test proxy: re-wrap test_ds.base with augment=True
    # so we get a newsprint-augmented synthetic test signal that correlates
    # with real-photo performance. See app/ml/train.py lines 138-158.
    test_aug_ds = AugmentedDataset(test_ds.base, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_aug_loader = DataLoader(
        test_aug_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    class_weights = torch.tensor(CLASS_WEIGHTS, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0
    stopped_at = EPOCHS

    print(
        f"\n[{config['name']}] "
        f"depth={config['depth']} "
        f"channels={config['channels']} "
        f"dropout={config['dropout']} "
        f"— {params:,} params"
    )

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                stopped_at = epoch
                break

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"  epoch {epoch:2d}/{EPOCHS}: "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                f"val_loss={val_loss:.4f}"
            )

    train_time = time.time() - start

    # Load best state for final evaluations
    assert best_state is not None
    model.load_state_dict(best_state)

    # 1a. Clean synthetic test split (historical comparison)
    synth_loss, synth_acc = evaluate(model, test_loader, criterion, device)

    # 1b. Augmented synthetic test split (L1 realistic proxy)
    synth_loss_aug, synth_acc_aug = evaluate(
        model, test_aug_loader, criterion, device,
    )

    # 2. Real-photo GT set (GT corners → isolates classifier quality)
    real = evaluate_real_photos(model, device, gt_cells)

    result = {
        "name": config["name"],
        "depth": config["depth"],
        "channels": config["channels"],
        "dropout": config["dropout"],
        "parameters": params,
        "best_val_acc": round(float(best_val_acc), 4),
        "best_val_loss": round(float(best_val_loss), 4),
        "synthetic_test_acc": round(float(synth_acc), 4),
        "synthetic_test_loss": round(float(synth_loss), 4),
        "synthetic_test_acc_aug": round(float(synth_acc_aug), 4),
        "synthetic_test_loss_aug": round(float(synth_loss_aug), 4),
        "real_filled_acc": real["filled_accuracy"],
        "real_empty_acc": real["empty_accuracy"],
        "real_missed_filled": real["missed_filled"],
        "real_wrong_digit": real["wrong_digit"],
        "real_hallucinated": real["hallucinated"],
        "epochs_run": stopped_at,
        "train_time_s": round(train_time, 1),
    }

    print(
        f"  DONE: "
        f"synth_clean={synth_acc:.4f} "
        f"synth_aug={synth_acc_aug:.4f} "
        f"real_filled={real['filled_accuracy']:.4f} "
        f"real_empty={real['empty_accuracy']:.4f} "
        f"| {train_time:.0f}s | epochs={stopped_at}"
    )
    return result


def build_configs() -> List[Dict[str, Any]]:
    """Build the full 3×3×3 config grid."""
    configs = []
    for depth in DEPTHS:
        for cname in CHANNEL_NAMES:
            channels = CHANNEL_SCHEMES[cname][depth]
            for dropout in DROPOUTS:
                name = f"d{depth}_c-{cname}_drop{dropout}"
                configs.append(
                    {
                        "name": name,
                        "depth": depth,
                        "channels": list(channels),
                        "dropout": dropout,
                    }
                )
    return configs


def save_results(results: List[Dict[str, Any]]) -> None:
    """Write results (plus protocol metadata) to JSON, crash-safe."""
    payload = {
        "protocol": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": f"Adam(lr={LR})",
            "scheduler": f"CosineAnnealingLR(T_max={EPOCHS})",
            "patience": PATIENCE,
            "seed": SEED,
            "hidden_fc": HIDDEN_FC,
            "loss": f"CrossEntropyLoss(weight={CLASS_WEIGHTS})",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "empty_threshold": EMPTY_THRESHOLD,
            "real_eval_mode": "GT corners (isolates OCR from detection)",
            "matches_production": (
                "v5.1 (2026-04-11) — class-weighted CE, threshold 0.50, "
                "v4.2 dataset (MNIST minus 0s + big-canvas PrintedDigit + "
                "rewritten EmptyCellDataset + Chars74K)"
            ),
            "notes": (
                "All configs trained under identical protocol on the same "
                "(MNIST-no-0 + PrintedDigit + EmptyCell + Chars74K) training "
                "data. Real-photo evaluation uses the 16-point ground-truth "
                "corners to warp each image, so differences in real-photo "
                "accuracy reflect classifier quality only, not detection "
                "quality. The augmented synthetic test split (test_aug) runs "
                "the same test samples through AugmentedDataset(augment=True) "
                "to give a realistic-distribution proxy — per the 2026-04-11 "
                "pipeline review, the clean test split is ~4x sharper than "
                "real photos and only correlates weakly with real-photo "
                "performance."
            ),
        },
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="SudokuCNN architecture ablation")
    parser.add_argument(
        "--only-baseline", action="store_true",
        help="Smoke test: only run the production baseline config",
    )
    parser.add_argument(
        "--max-configs", type=int, default=None,
        help="Run only the first N configs (for debugging)",
    )
    parser.add_argument(
        "--dropout-only", type=float, default=None,
        help=(
            "Filter to configs with exactly this dropout value "
            "(e.g. 0.3 → 9-config depth × channels diagonal)"
        ),
    )
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="Discard existing results file and start fresh",
    )
    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Build config grid
    configs = build_configs()
    if args.only_baseline:
        configs = [c for c in configs if c["name"] == "d3_c-medium_drop0.3"]
    if args.dropout_only is not None:
        configs = [c for c in configs if c["dropout"] == args.dropout_only]
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    # Resume: skip configs that already have results
    existing_results: List[Dict[str, Any]] = []
    if OUTPUT_PATH.exists() and not args.force_rerun:
        try:
            existing_results = json.loads(OUTPUT_PATH.read_text()).get("results", [])
            done = {r["name"] for r in existing_results}
            configs = [c for c in configs if c["name"] not in done]
            if existing_results:
                print(
                    f"Resuming: {len(existing_results)} configs already done, "
                    f"{len(configs)} remaining"
                )
        except Exception:
            existing_results = []

    print(f"Device: {device}")
    print(f"Configs to run: {len(configs)}")

    # Pre-load and pre-process all GT cells once (used by every config's eval)
    print(f"Loading ground-truth cells from {IMAGES_DIR}...")
    gt_cells = load_gt_cells()
    print(f"  {len(gt_cells)} ground-truth images loaded")

    results = list(existing_results)
    overall_start = time.time()

    for i, config in enumerate(configs, 1):
        print(f"\n===== [{i}/{len(configs)}] =====")
        try:
            result = train_one_config(config, device, gt_cells)
            results.append(result)
            save_results(results)
        except KeyboardInterrupt:
            print(f"\nInterrupted after {i-1}/{len(configs)} configs — results saved.")
            break
        except Exception as e:
            print(f"FAILED for {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - overall_start
    print(
        f"\n===== COMPLETE: {len(results)} configs "
        f"in {total_time / 60:.1f} minutes ====="
    )
    print(f"Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
