"""
MNIST-based digit dataset for Sudoku OCR.

Downloads MNIST via torchvision and remaps labels for Sudoku use:
- MNIST digit 0 images are relabeled to class 0 (empty cell)
- MNIST digits 1-9 keep their labels
- Synthetic empty cells (blank/noisy) are added to class 0 to teach
  the model what real empty Sudoku cells look like.

Augmentations simulate real camera capture: rotation, affine warp,
noise, blur, and brightness variation.
"""

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import datasets, transforms

# Where to cache the downloaded MNIST data
DATA_DIR = "data/mnist"


class EmptyCellDataset(Dataset):
    """Synthetic empty Sudoku cells: blank, noisy, gradient, grid-line remnants.

    These augment the MNIST '0' class so the model learns that empty cells
    aren't just digit-zero but also blank/textured images.
    """

    def __init__(self, count: int = 5000, size: int = 28, seed: int = 42):
        self.size = size
        self.count = count
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self._generate(idx)
        tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        return tensor, 0

    def _generate(self, idx: int) -> np.ndarray:
        choice = idx % 4
        s = self.size
        if choice == 0:
            return np.zeros((s, s), dtype=np.uint8)
        elif choice == 1:
            level = self.rng.randint(5, 30)
            return self.rng.randint(0, level, (s, s)).astype(np.uint8)
        elif choice == 2:
            end = self.rng.randint(10, 40)
            grad = np.linspace(0, end, s, dtype=np.float32)
            if idx % 2 == 0:
                return np.tile(grad, (s, 1)).astype(np.uint8)
            return np.tile(grad.reshape(-1, 1), (1, s)).astype(np.uint8)
        else:
            img = np.zeros((s, s), dtype=np.uint8)
            t = self.rng.randint(1, 3)
            b = self.rng.randint(30, 80)
            for edge in range(4):
                if self.rng.random() < 0.5:
                    if edge == 0:
                        img[:t, :] = b
                    elif edge == 1:
                        img[-t:, :] = b
                    elif edge == 2:
                        img[:, :t] = b
                    else:
                        img[:, -t:] = b
            return img


class AugmentedDataset(Dataset):
    """Wraps a dataset and applies augmentations on the fly."""

    def __init__(self, base: Dataset, augment: bool = True):
        self.base = base
        self.augment = augment

        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomRotation(15, fill=0),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.08, 0.08),
                        scale=(0.85, 1.15),
                        shear=(-10, 10),
                        fill=0,
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base[idx]

        # img may be a tensor (1,28,28) or PIL — normalize to tensor
        if isinstance(img, torch.Tensor):
            if self.augment and self.transform:
                # Convert back to uint8 numpy for ToPILImage
                np_img = (img.squeeze(0).numpy() * 255).astype(np.uint8)
                np_img = self._apply_noise(np_img)
                img = self.transform(np_img)
            return img, label
        else:
            # PIL Image path
            tensor = transforms.ToTensor()(img)
            return tensor, label

    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        if random.random() < 0.4:
            noise = np.random.normal(0, random.uniform(5, 15), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            shift = random.randint(-20, 20)
            img = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        if random.random() < 0.25:
            import cv2

            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return img


def _load_mnist(train: bool) -> Dataset:
    """Download and load MNIST, returning tensors in (1, 28, 28) float [0,1]."""
    return datasets.MNIST(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )


def create_datasets(
    empty_cell_count: int = 5000,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Create train/val/test datasets from MNIST + synthetic empty cells.

    - Train: MNIST train split (60K) + synthetic empty cells, with augmentation
    - Val: 10% held out from train, no augmentation
    - Test: MNIST test split (10K) + synthetic empty cells, no augmentation

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    rng = torch.Generator().manual_seed(seed)

    # Load MNIST
    mnist_train = _load_mnist(train=True)
    mnist_test = _load_mnist(train=False)

    # Add synthetic empty cells to train
    empty_train = EmptyCellDataset(count=empty_cell_count, seed=seed)
    full_train = ConcatDataset([mnist_train, empty_train])

    # Split train into train + val (90/10)
    n = len(full_train)
    val_size = n // 10
    train_size = n - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [train_size, val_size], generator=rng
    )

    # Wrap with augmentation
    train_ds = AugmentedDataset(train_subset, augment=True)
    val_ds = AugmentedDataset(val_subset, augment=False)

    # Test set: MNIST test + some empty cells
    empty_test = EmptyCellDataset(count=1000, seed=seed + 1)
    test_ds = AugmentedDataset(ConcatDataset([mnist_test, empty_test]), augment=False)

    return train_ds, val_ds, test_ds
