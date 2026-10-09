"""Visible tests for training: build_model(), unfreeze_last_blocks(), make_optimizer() and evaluate_epoch().
Random weights and synthetic images: nothing is downloaded.

Run: pytest tests/visible/test_train.py -v
"""
from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from snapclassify.data import synthetic_pets
from snapclassify.train import build_model, evaluate_epoch, make_optimizer, unfreeze_last_blocks


def test_build_model_freezes_the_backbone_and_replaces_the_head():
    model = build_model(37, pretrained=False)
    assert model.classifier[-1].out_features == 37
    assert not any(p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 37)


def test_unfreeze_last_blocks_and_the_optimizer_groups():
    model = build_model(10, pretrained=False)
    unfreeze_last_blocks(model)
    trainable = [any(p.requires_grad for p in block.parameters()) for block in model.features]
    assert trainable == [False] * 11 + [True, True]
    opt = make_optimizer(model, lr=1e-3)
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([1e-3, 1e-4])       # the backbone at a tenth
    assert len(opt.param_groups[1]["params"]) == len(list(model.features[-2:].parameters()))


def test_evaluate_epoch_weights_every_image_equally():
    data = synthetic_pets(num_classes=4, per_class=5)                  # 20 images: batches of 6, 6, 6 and 2
    always_first = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 4))
    nn.init.zeros_(always_first[1].weight)
    with torch.no_grad():
        always_first[1].bias.copy_(torch.tensor([2.0, 0.0, 0.0, 0.0]))
    loss, acc = evaluate_epoch(always_first, DataLoader(data, batch_size=6), nn.CrossEntropyLoss())
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    per_class = [nn.functional.cross_entropy(logits, torch.tensor([c])).item() for c in range(4)]
    assert acc == pytest.approx(0.25)                                   # it always says class 0
    assert loss == pytest.approx(sum(per_class) / 4, rel=1e-5)
    assert not always_first.training
