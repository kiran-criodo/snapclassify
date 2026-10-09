"""The data: Oxford-IIIT Pet, its splits and transforms, and a synthetic stand-in for tests. Given.

Oxford-IIIT Pet has 37 breeds (12 cats, 25 dogs), about 200 photos each, split by its authors into "trainval"
(3,680 photos) and "test" (3,669). torchvision downloads it (about 800 MB) into --data the first time. It is
licensed CC BY-SA 4.0: never commit it (data/pets/ is git-ignored) and credit it wherever the model is used.

    train / val   a seeded 85/15 split of "trainval": train to fit, val to pick the best epoch
    test          untouched until you measure the trained model
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

from snapclassify.constants import CROP, MEAN, RESIZE, STD

DATA_DIR = Path("data/pets")


def eval_transform() -> transforms.Compose:
    """Resize the shorter side to 256, keep the centre 224 x 224, scale to [0, 1], normalise. predict.preprocess()
    must match this exactly."""
    return transforms.Compose([transforms.Resize(RESIZE), transforms.CenterCrop(CROP), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def train_transform() -> transforms.Compose:
    """Light augmentation: a random crop of most of the photo, a mirror image, a little colour change. A pet
    photographed from a different distance or angle is still the same breed."""
    return transforms.Compose([transforms.RandomResizedCrop(CROP, scale=(0.6, 1.0)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def _subset_indices(ds: datasets.OxfordIIITPet, subset: int | None) -> list[int]:
    labels = ds._labels                                   # the class index of every photo, in file order
    return [i for i, y in enumerate(labels) if subset is None or y < subset]


def load_pets(root: str | Path = DATA_DIR, subset: int | None = None, download: bool = True, seed: int = 0,
              val_fraction: float = 0.15) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    """(train, val, test, class_names). subset=10 keeps the first 10 breeds (by class index) for a CPU run."""
    root = Path(root)
    kw = {"root": str(root), "target_types": "category", "download": download}
    trainval_aug = datasets.OxfordIIITPet(split="trainval", transform=train_transform(), **kw)
    trainval_eval = datasets.OxfordIIITPet(split="trainval", transform=eval_transform(), **kw)
    test = datasets.OxfordIIITPet(split="test", transform=eval_transform(), **kw)
    idx = _subset_indices(trainval_eval, subset)
    random.Random(seed).shuffle(idx)
    n_val = int(len(idx) * val_fraction)
    val_idx, train_idx = sorted(idx[:n_val]), sorted(idx[n_val:])
    names = trainval_eval.classes[:subset] if subset else list(trainval_eval.classes)
    return (Subset(trainval_aug, train_idx), Subset(trainval_eval, val_idx),
            Subset(test, _subset_indices(test, subset)), names)


def load_test_photos(root: str | Path = DATA_DIR, subset: int | None = None) -> tuple[Dataset, list[str]]:
    """The test split as (PIL image, label) pairs, for the Gemini comparison and the CPU latency check."""
    ds = datasets.OxfordIIITPet(root=str(root), split="test", target_types="category", download=True)
    names = ds.classes[:subset] if subset else list(ds.classes)
    return Subset(ds, _subset_indices(ds, subset)), names


def synthetic_pets(num_classes: int = 4, per_class: int = 8, size: int = 64, seed: int = 0) -> TensorDataset:
    """Normalised image tensors where each class has its own colour and stripe pattern plus noise: learnable in an
    epoch or two, so the training loop and the tests run in seconds without the dataset."""
    g = torch.Generator().manual_seed(seed)                # the noise depends on the seed ...
    xs, ys = [], []
    for c in range(num_classes):                            # ... the class's look doesn't
        colour = torch.rand(3, 1, 1, generator=torch.Generator().manual_seed(1000 + c)) * 2 - 1
        stripes = (torch.arange(size) // (2 + c) % 2).float().view(1, 1, size) * 0.8
        for _ in range(per_class):
            xs.append(colour + stripes + 0.3 * torch.randn(3, size, size, generator=g))
            ys.append(c)
    return TensorDataset(torch.stack(xs), torch.tensor(ys))


def synthetic_names(num_classes: int) -> list[str]:
    return [f"Breed {chr(65 + i)}" for i in range(num_classes)]


def make_loaders(train: Dataset, val: Dataset, batch_size: int = 32, workers: int = 0) -> tuple[DataLoader, DataLoader]:
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=len(train) > batch_size),
            DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=workers))
