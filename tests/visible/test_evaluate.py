"""Visible tests for evaluating: topk_accuracy(), per_class_accuracy(), most_confused() and compare_table().
Fixture numbers only: no dataset, no API calls.

Run: pytest tests/visible/test_evaluate.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from snapclassify.evaluate import compare_table, most_confused, per_class_accuracy, topk_accuracy

NAMES = ["Abyssinian", "Beagle", "Bengal", "Boxer"]
LOGITS = np.array([[3.0, 1.0, 0.0, -1.0],     # Abyssinian: right
                   [0.5, 2.0, 1.0, 0.0],      # Bengal, called Beagle; Bengal is 2nd, so in the top 3
                   [0.0, 0.1, 0.2, 3.0],      # Boxer: right
                   [2.0, 0.0, 1.0, 0.5],      # Beagle, called Abyssinian; Beagle is last, not in the top 3
                   [0.0, 4.0, 1.0, 2.0]])     # Beagle: right
TARGETS = np.array([0, 2, 3, 1, 1])


def test_topk_accuracy():
    assert topk_accuracy(LOGITS, TARGETS, k=1) == pytest.approx(3 / 5)
    assert topk_accuracy(LOGITS, TARGETS, k=3) == pytest.approx(4 / 5)


def test_per_class_accuracy_and_most_confused():
    preds = LOGITS.argmax(axis=1)
    assert per_class_accuracy(preds, TARGETS, 4) == pytest.approx([1.0, 0.5, 0.0, 1.0])
    assert most_confused(preds, TARGETS, NAMES) == [("Beagle", "Abyssinian", 1), ("Bengal", "Beagle", 1)]


def test_compare_table_same_photos_both_models():
    cnn = [{"target": t, "label": p, "latency_s": s} for t, p, s in
           [("Beagle", "Beagle", 0.02), ("Boxer", "Beagle", 0.03), ("Bengal", "Bengal", 0.01), ("Boxer", "Boxer", 0.02)]]
    llm = [{"target": t, "label": p, "latency_s": s, "input_tokens": 1100, "output_tokens": 50} for t, p, s in
           [("Beagle", "beagle", 1.2), ("Boxer", "Boxer", 1.5), ("Bengal", None, 2.0), ("Boxer", "Boxer", 1.1)]]
    cnn_row, llm_row = compare_table(cnn, llm, "gemini-3.8-flash", {"input": 0.75, "output": 3.75})
    assert (cnn_row["model"], cnn_row["n"], cnn_row["cost_per_1k"]) == ("snapclassify (CPU)", 4, 0.0)
    assert cnn_row["accuracy"] == pytest.approx(0.75) and cnn_row["p50_s"] == pytest.approx(0.02)
    assert llm_row["model"] == "gemini-3.8-flash"
    assert llm_row["accuracy"] == pytest.approx(0.75)                  # "beagle" counts; None doesn't
    assert llm_row["p50_s"] == pytest.approx(1.35)
    assert llm_row["cost_per_1k"] == pytest.approx(1.0125)             # (1100 x 0.75 + 50 x 3.75) / 1e6, x 1,000
