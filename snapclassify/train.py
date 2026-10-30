"""Transfer learning: an ImageNet MobileNetV3-Small with a new head for the breeds.

    build_model()          load the pretrained network, freeze the backbone, replace the last layer
    unfreeze_last_blocks() make the last feature blocks trainable for the fine-tuning phase
    make_optimizer()       the head at lr, the unfrozen backbone at a tenth of it
    evaluate_epoch()       validation loss and accuracy
    train_one_epoch(), fit()   the head-only phase, then fine-tuning; keeps the best epoch            given

    python -m snapclassify.train --synthetic           # seconds on a CPU: checks your code, no download
    python -m snapclassify.train --subset 10           # 10 breeds on a CPU (downloads the dataset once)
    python -m snapclassify.train --epochs-finetune 3   # all 37 breeds: use a GPU (notebooks/train_colab.ipynb)
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

MODELS_DIR = Path("models")
WEIGHTS_FILE = "snapclassify.pt"
LABELS_FILE = "labels.json"


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Small with every backbone (model.features) parameter frozen and a new last layer."""
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def unfreeze_last_blocks(model: nn.Module, n: int = 2) -> None:
    """The fine-tuning phase: make the last n blocks of model.features trainable; the earlier blocks stay frozen."""
    for p in model.features[len(model.features) - n:].parameters():
        p.requires_grad = True


def make_optimizer(model: nn.Module, lr: float = 1e-3, backbone_factor: float = 0.1) -> torch.optim.Optimizer:
    """AdamW over the trainable parameters only: the head (model.classifier) at lr, any trainable backbone
    parameters (model.features) at lr * backbone_factor."""
    head = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    groups = [{"params": head, "lr": lr}]
    if backbone:
        groups.append({"params": backbone, "lr": lr * backbone_factor})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def evaluate_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str = "cpu") -> tuple[float, float]:
    """(mean loss per image, accuracy) over the loader, without updating the model."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


def _frozen_batchnorm_to_eval(model: nn.Module) -> None:
    """model.train() also switches frozen batch-norm layers to training mode, where they would keep updating their
    running statistics from pet photos and quietly change the frozen backbone. Put them back in eval mode. Given."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                    device: str = "cpu") -> tuple[float, float]:
    """One pass over the training data: (mean loss, accuracy). Given."""
    model.train()
    _frozen_batchnorm_to_eval(model)
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, class_names: list[str],
        out_dir: str | Path = MODELS_DIR, *, epochs_head: int = 3, epochs_finetune: int = 2, lr: float = 1e-3,
        device: str = "cpu", log: Callable[[str], None] = print) -> list[dict]:
    """The head-only phase trains the new head with the backbone frozen; the fine-tuning phase then unfreezes the
    last blocks at a lower learning rate.
    After every epoch the validation accuracy is measured, and the best epoch's weights are the ones saved, with
    labels.json and history.json, in out_dir. Returns the history. Given."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    history: list[dict] = []
    best_acc, best_state = -1.0, None
    for phase, epochs in (("head", epochs_head), ("finetune", epochs_finetune)):
        if phase == "finetune":
            unfreeze_last_blocks(model)
        optimizer = make_optimizer(model, lr)
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)
            row = {"phase": phase, "epoch": epoch, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
                   "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
                   "seconds": round(time.perf_counter() - t0, 1)}
            history.append(row)
            log(f"{phase:8} epoch {epoch}: train loss {train_loss:.3f} acc {train_acc:.3f} · "
                f"val loss {val_loss:.3f} acc {val_acc:.3f} · {row['seconds']}s")
            if val_acc > best_acc:
                best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(best_state, out / WEIGHTS_FILE)
    (out / LABELS_FILE).write_text(json.dumps(class_names, indent=1), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    log(f"best val acc {best_acc:.3f}; saved {out / WEIGHTS_FILE}, {out / LABELS_FILE}, {out / 'history.json'}")
    return history


def calibrate_batchnorm(model: nn.Module, loader: DataLoader) -> None:
    """Only for random weights (--synthetic): a randomly initialised MobileNetV3 in eval mode shrinks every image to
    almost the same features, so nothing can learn from them. One pass over the data in training mode, without
    gradients, sets every batch-norm layer's running statistics from real activations. Pretrained weights don't
    need this. Given."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                     # a plain average over the pass
    model.train()
    with torch.no_grad():
        for images, _ in loader:
            model(images)
    for m in bns:
        m.momentum = 0.1


def load_trained(model_dir: str | Path = MODELS_DIR, device: str = "cpu") -> tuple[nn.Module, list[str]]:
    """The saved model (in eval mode) and its class names. Given."""
    model_dir = Path(model_dir)
    names = json.loads((model_dir / LABELS_FILE).read_text(encoding="utf-8"))
    model = build_model(len(names), pretrained=False)
    model.load_state_dict(torch.load(model_dir / WEIGHTS_FILE, map_location=device))
    return model.to(device).eval(), names


def main(argv: list[str] | None = None) -> int:
    from snapclassify.data import DATA_DIR, load_pets, make_loaders, synthetic_names, synthetic_pets
    ap = argparse.ArgumentParser(description="Train SnapClassify's breed model.")
    ap.add_argument("--data", default=str(DATA_DIR), help="where the dataset is (downloaded there if missing)")
    ap.add_argument("--subset", type=int, help="the first N breeds only (10 for a CPU run)")
    ap.add_argument("--synthetic", action="store_true", help="generated images: checks the code, no download")
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-finetune", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(MODELS_DIR))
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.synthetic:
        train, val = synthetic_pets(4, 24, seed=0), synthetic_pets(4, 6, seed=1)
        names, pretrained, workers, batch_size = synthetic_names(4), False, 0, 16
        print("synthetic images and random weights: val accuracy should climb well above 0.25 (chance)")
    else:
        train, val, _, names = load_pets(a.data, a.subset)
        pretrained, workers, batch_size = True, a.workers, a.batch_size
    print(f"{len(names)} classes · {len(train)} train · {len(val)} val · device {device}")
    train_loader, val_loader = make_loaders(train, val, batch_size, workers)
    model = build_model(len(names), pretrained=pretrained)
    if a.synthetic:
        calibrate_batchnorm(model, train_loader)
    fit(model, train_loader, val_loader, names, a.out, epochs_head=a.epochs_head, epochs_finetune=a.epochs_finetune,
        lr=a.lr, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
