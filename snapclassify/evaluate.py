"""Honest evaluation: the test split, per-class numbers, the confusions, and the CNN against Gemini.

    topk_accuracy()        the share of photos whose breed is in the model's top k
    per_class_accuracy()   accuracy per breed (None for a breed with no test photos)
    most_confused()        the (true breed, predicted breed, count) pairs the model mixes up most
    compare_table()        the CNN and the LLM on the same photos: accuracy, p50 latency, cost per 1,000
    collect_logits(), threshold_table(), write_report()                                                  given

    python -m snapclassify.evaluate --synthetic                # checks the code on generated images
    python -m snapclassify.evaluate --subset 10                # the test split, 10 breeds
    python -m snapclassify.evaluate --subset 10 --compare 50   # + Gemini on 50 of those photos (GEMINI_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPORT = Path("reports/eval.md")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def _name(s) -> str | None:
    return None if s is None else " ".join(str(s).replace("_", " ").split()).casefold()


def topk_accuracy(logits: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    """logits: (n, classes) scores; targets: (n,) class indices."""
    top = np.argsort(-logits, axis=1)[:, :k]
    return float((top == targets[:, None]).any(axis=1).mean())


def per_class_accuracy(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> list[float | None]:
    """preds and targets: (n,) class indices. One entry per class, in class order."""
    out: list[float | None] = []
    for c in range(num_classes):
        rows = targets == c
        out.append(float((preds[rows] == c).mean()) if rows.any() else None)
    return out


def most_confused(preds: np.ndarray, targets: np.ndarray, class_names: list[str], n: int = 5) -> list[tuple[str, str, int]]:
    """The n most frequent mistakes as (true breed, predicted breed, count)."""
    counts = Counter((class_names[t], class_names[p]) for t, p in zip(targets.tolist(), preds.tolist()) if t != p)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return [(true, pred, count) for (true, pred), count in ranked[:n]]


def compare_table(cnn_rows: list[dict], llm_rows: list[dict], llm_model: str,
                  prices: dict[str, float]) -> list[dict]:
    """Both models on the same photos. A row: {"target": breed, "label": breed or None, "latency_s", and for the LLM
    "input_tokens", "output_tokens" and "error" (None, or why the call failed)}. Returns one dict per model:
    {"model", "n", "accuracy", "p50_s", "cost_per_1k"}."""
    def side(model: str, rows: list[dict], cost_per_1k: float | None) -> dict:
        right = sum(r["label"] is not None and _name(r["label"]) == _name(r["target"]) for r in rows)
        return {"model": model, "n": len(rows), "accuracy": right / len(rows),
                "p50_s": float(np.median([r["latency_s"] for r in rows])), "cost_per_1k": cost_per_1k}

    costs = [(r["input_tokens"] * prices["input"] + r["output_tokens"] * prices["output"]) / 1_000_000
             for r in llm_rows if r.get("error") is None]
    llm_cost = 1000 * float(np.mean(costs)) if costs else None      # None: every call failed, so no price to report
    return [side("snapclassify (CPU)", cnn_rows, 0.0), side(llm_model, llm_rows, llm_cost)]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def threshold_table(probs: np.ndarray, targets: np.ndarray, thresholds=THRESHOLDS) -> list[dict]:
    """For each threshold: the share of photos answered "not sure" and the top-1 accuracy on the rest. The app's
    "not sure" threshold is chosen from this table. Given."""
    top, conf = probs.argmax(axis=1), probs.max(axis=1)
    rows = []
    for t in thresholds:
        sure = conf >= t
        rows.append({"threshold": t, "not_sure": float(1 - sure.mean()),
                     "accuracy_when_sure": float((top[sure] == targets[sure]).mean()) if sure.any() else None})
    return rows


def collect_logits(model, loader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Every test photo through the model: (logits, targets) as NumPy arrays. Given."""
    import torch
    model.eval()
    out, ys = [], []
    with torch.no_grad():
        for images, labels in loader:
            out.append(model(images.to(device)).cpu().numpy())
            ys.append(labels.numpy())
    return np.concatenate(out), np.concatenate(ys)


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}%"


def _usd(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def write_report(path: Path, *, names: list[str], logits: np.ndarray, targets: np.ndarray,
                 comparison: list[dict] | None = None, source: str = "") -> None:
    """reports/eval.md, written afresh on every run. Given."""
    preds = logits.argmax(axis=1)
    per_class = per_class_accuracy(preds, targets, len(names))
    lines = ["# SnapClassify evaluation", "", f"Written by `python -m snapclassify.evaluate` ({source}); rerun it "
             "rather than editing this file.", "",
             f"Test photos: {len(targets)} · breeds: {len(names)} · top-1 {_pct(topk_accuracy(logits, targets, 1))}"
             f" · top-3 {_pct(topk_accuracy(logits, targets, 3))}", "",
             "## \"Not sure\" threshold", "", "| threshold | answered \"not sure\" | top-1 when sure |", "|---|---|---|"]
    lines += [f"| {r['threshold']} | {_pct(r['not_sure'])} | {_pct(r['accuracy_when_sure'])} |"
              for r in threshold_table(softmax(logits), targets)]
    lines += ["", "## Per breed (worst first)", "", "| breed | test photos | accuracy |", "|---|---|---|"]
    order = sorted(range(len(names)), key=lambda c: (per_class[c] if per_class[c] is not None else 2, names[c]))
    lines += [f"| {names[c]} | {int((targets == c).sum())} | {_pct(per_class[c])} |" for c in order]
    lines += ["", "## Most confused (true -> predicted)", "", "| true breed | predicted as | photos |", "|---|---|---|"]
    lines += [f"| {t} | {p} | {n} |" for t, p, n in most_confused(preds, targets, names)]
    if comparison:
        lines += ["", "## The CNN against a multimodal LLM (same photos)", "",
                  "| model | photos | top-1 | p50 latency (s) | $ per 1,000 photos |", "|---|---|---|---|---|"]
        lines += [f"| {r['model']} | {r['n']} | {_pct(r['accuracy'])} | {r['p50_s']:.3f} | {_usd(r['cost_per_1k'])} |"
                  for r in comparison]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _compare(model, names: list[str], data: str, subset: int | None, n: int, pace: float) -> list[dict]:
    """The CNN on a CPU and Gemini on the same n test photos (a fixed sample). Given."""
    import torch

    from snapclassify.data import eval_transform, load_test_photos
    from snapclassify.llm_vision import MODEL, PRICES, classify_photo, make_client
    photos, _ = load_test_photos(data, subset)
    sample = sorted(random.Random(0).sample(range(len(photos)), min(n, len(photos))))
    model = model.cpu().eval()
    tf, client = eval_transform(), make_client()
    cnn_rows, llm_rows = [], []
    for i, idx in enumerate(sample, 1):
        image, y = photos[idx]
        t0 = time.perf_counter()
        with torch.no_grad():
            label = names[int(model(tf(image).unsqueeze(0)).argmax(1))]
        cnn_rows.append({"target": names[y], "label": label, "latency_s": time.perf_counter() - t0})
        llm_rows.append({"target": names[y], **classify_photo(client, image, names)})
        print(f"  {i}/{len(sample)} {names[y]}: cnn {label} · llm {llm_rows[-1]['label']}")
        if pace:
            time.sleep(pace)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/compare.json").write_text(json.dumps({"cnn": cnn_rows, "llm": llm_rows}, indent=1), encoding="utf-8")
    return compare_table(cnn_rows, llm_rows, MODEL, PRICES)


def main(argv: list[str] | None = None) -> int:
    from torch.utils.data import DataLoader

    from snapclassify.data import DATA_DIR, load_pets, synthetic_names, synthetic_pets
    from snapclassify.train import MODELS_DIR, load_trained
    ap = argparse.ArgumentParser(description="Evaluate SnapClassify on the test split.")
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--model", default=str(MODELS_DIR), help="the folder with snapclassify.pt and labels.json")
    ap.add_argument("--subset", type=int, help="the same --subset you trained with")
    ap.add_argument("--synthetic", action="store_true", help="generated images (after train --synthetic)")
    ap.add_argument("--compare", type=int, metavar="N", help="also run Gemini on N test photos (needs a key)")
    ap.add_argument("--pace", type=float, default=6.5, help="seconds between Gemini calls (free-tier limits)")
    ap.add_argument("--out", default=str(REPORT))
    a = ap.parse_args(argv)
    if a.compare and not a.synthetic:
        from snapclassify.env import load_env
        load_env()
        if not os.environ.get("GEMINI_API_KEY"):
            print("--compare needs your Gemini key: put it after GEMINI_API_KEY= in .env, then run this again")
            return 1
    model, names = load_trained(a.model)
    if a.synthetic:
        test, source = synthetic_pets(len(names), 10, seed=2), "synthetic images"
    else:
        _, _, test, _ = load_pets(a.data, a.subset)
        source = f"Oxford-IIIT Pet test split{f', first {a.subset} breeds' if a.subset else ''}"
    if a.synthetic and names != synthetic_names(len(names)):
        print("warning: this model wasn't trained with --synthetic")
    logits, targets = collect_logits(model, DataLoader(test, batch_size=64))
    comparison = None
    if a.compare:
        if a.synthetic:
            print("--compare needs real photos: drop --synthetic")
            return 1
        from snapclassify.env import load_env
        load_env()
        comparison = _compare(model, names, a.data, a.subset, a.compare, a.pace)
    write_report(Path(a.out), names=names, logits=logits, targets=targets, comparison=comparison, source=source)
    print(f"top-1 {topk_accuracy(logits, targets, 1):.3f} · top-3 {topk_accuracy(logits, targets, 3):.3f} "
          f"on {len(targets)} photos; wrote {a.out}")
    for r in comparison or []:
        print(f"{r['model']:22} top-1 {r['accuracy']:.3f} · p50 {r['p50_s']:.3f}s · ${_usd(r['cost_per_1k'])} per 1,000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
