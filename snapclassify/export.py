"""Export to ONNX: one file the app can run with ONNX Runtime, no PyTorch needed.

    export_onnx()   torch.onnx.export with a dynamic batch size, then a parity check against PyTorch      TODO

    python -m snapclassify.export --threshold 0.5          # models/ -> app/model/ (model.onnx, labels.json, config.json)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

APP_MODEL_DIR = Path("app/model")


def export_onnx(model: nn.Module, path: str | Path, image_size: int = 224, rtol: float = 1e-3,
                atol: float = 1e-4) -> float:
    """Write the model to path as ONNX (input "image", output "logits", any batch size) and return the largest
    difference between ONNX Runtime's and PyTorch's logits on the same input."""
    # TODO: CRIO_TASK_MODULE_SNAP
    # 1. model.eval() (dropout off, batch norm frozen): the exported graph is the inference graph.
    # 2. torch.onnx.export(model, (example,), str(path), input_names=["image"], output_names=["logits"],
    #        dynamic_shapes=({0: torch.export.Dim("batch")},), external_data=False)
    #    with example = torch.randn(2, 3, image_size, image_size). Dim("batch") makes dimension 0 dynamic, so the
    #    app can send one photo and a batch job 64; external_data=False keeps the weights inside the one file.
    # 3. Parity: a new input with a different batch size, check = torch.randn(3, 3, image_size, image_size);
    #    expected = model(check) under torch.no_grad(); got = the output of
    #    onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": check.numpy()})
    # 4. The two runtimes round differently, so compare like np.allclose(got, expected, rtol=rtol, atol=atol): raise
    #    ValueError if they aren't close; otherwise return the largest absolute difference as a float.
    # Test with: pytest tests/visible/test_ship.py -k export
    raise NotImplementedError("TODO export_onnx(): ONNX with a dynamic batch size, then the parity check")


def main(argv: list[str] | None = None) -> int:
    from snapclassify.constants import CROP
    from snapclassify.train import LABELS_FILE, MODELS_DIR, load_trained
    ap = argparse.ArgumentParser(description="Export the trained model for the app.")
    ap.add_argument("--model", default=str(MODELS_DIR))
    ap.add_argument("--out", default=str(APP_MODEL_DIR))
    ap.add_argument("--threshold", type=float, required=True,
                    help='below this top confidence the app says "not sure": choose it from reports/eval.md')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, names = load_trained(a.model)
    diff = export_onnx(model, out / "model.onnx", CROP)
    shutil.copy(Path(a.model) / LABELS_FILE, out / "labels.json")
    (out / "config.json").write_text(json.dumps({"threshold": a.threshold, "breeds": len(names)}, indent=1) + "\n",
                                     encoding="utf-8")
    size = (out / "model.onnx").stat().st_size / 1e6
    print(f"wrote {out}/model.onnx ({size:.1f} MB), labels.json ({len(names)} breeds), config.json "
          f"(threshold {a.threshold}); ONNX matches PyTorch within {diff:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
