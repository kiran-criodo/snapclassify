"""Shared test setup: offline and deterministic. Tests use synthetic images and randomly initialised weights, so
nothing is downloaded: no dataset, no pretrained weights, no API calls."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(autouse=True)
def _seeded(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def photo() -> Image.Image:
    """A 457 x 301 RGB 'photo' of random pixels: an odd size, so resizing and cropping are really tested."""
    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((301, 457, 3)) * 255).astype(np.uint8))


class FakeSession:
    """Stands in for an ONNX Runtime session: returns fixed logits and records the input it was given."""

    def __init__(self, logits):
        self.logits = np.asarray([logits], dtype=np.float32)
        self.inputs = []

    def run(self, output_names, feed):
        self.inputs.append(feed)
        return [self.logits]


@pytest.fixture
def fake_session():
    return FakeSession
