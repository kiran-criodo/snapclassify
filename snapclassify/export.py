"""Export to ONNX: one file the app can run with ONNX Runtime, no PyTorch needed.

    export_onnx()   torch.onnx.export with a dynamic batch size, then a parity check against PyTorch

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
    import onnxruntime as ort
    model.eval()
    example = torch.randn(2, 3, image_size, image_size)
    torch.onnx.export(model, (example,), str(path), input_names=["image"], output_names=["logits"],
                      dynamic_shapes=({0: torch.export.Dim("batch")},), external_data=False)
    check = torch.randn(3, 3, image_size, image_size)                # a different batch size from the export
    with torch.no_grad():
        expected = model(check).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"image": check.numpy()})[0]
    if not np.allclose(got, expected, rtol=rtol, atol=atol):
        raise ValueError(f"ONNX and PyTorch disagree: largest difference {np.abs(got - expected).max():.1e}")
    return float(np.abs(got - expected).max())


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
