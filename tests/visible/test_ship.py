"""Visible tests for shipping: export_onnx(), preprocess() and predict().

Run: pytest tests/visible/test_ship.py -v
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from snapclassify.data import eval_transform, make_loaders, synthetic_pets
from snapclassify.export import export_onnx
from snapclassify.predict import predict, preprocess
from snapclassify.train import build_model, calibrate_batchnorm


def test_export_onnx_matches_pytorch_with_any_batch_size(tmp_path):
    model = build_model(5, pretrained=False)
    calibrate_batchnorm(model, make_loaders(synthetic_pets(5, 4), synthetic_pets(5, 1))[0])   # non-trivial outputs
    diff = export_onnx(model, tmp_path / "model.onnx", image_size=64)
    assert 0.0 <= diff < 1.0
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "image" and not isinstance(session.get_inputs()[0].shape[0], int)
    for batch in (1, 4):
        out = session.run(None, {"image": np.random.randn(batch, 3, 64, 64).astype(np.float32)})[0]
        assert out.shape == (batch, 5)


def test_preprocess_matches_the_evaluation_transform(photo):
    x = preprocess(photo)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    np.testing.assert_allclose(x, eval_transform()(photo).numpy()[None], atol=1e-5)


def test_predict_returns_the_top_three_or_not_sure(photo, fake_session):
    session = fake_session([2.0, 0.5, 1.0, -1.0])
    labels = ["Abyssinian", "Beagle", "Bengal", "Boxer"]
    r = predict(photo, session, labels, threshold=0.5)
    assert session.inputs[0]["image"].shape == (1, 3, 224, 224)       # the photo went through preprocess()
    assert [t["breed"] for t in r["top"]] == ["Abyssinian", "Bengal", "Beagle"]
    assert r["top"][0]["confidence"] == pytest.approx(0.6095, abs=1e-4)
    assert (r["suggestion"], r["not_sure"]) == ("Abyssinian", False)
    r = predict(photo, session, labels, threshold=0.7)
    assert (r["suggestion"], r["not_sure"], len(r["top"])) == (None, True, 3)
