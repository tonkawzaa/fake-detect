"""
Phase 5: measure ensemble accuracy on data/eval/ (built by fetch_eval_set.py)
and write app/calibration.json.

Critically, this does NOT fit ensemble weights/Platt scaling and then report
accuracy on the same images -- that would be training-set performance under
an "accuracy" label, exactly the conflation the project's plan opens by
warning against. We fit inside stratified K-fold CV and report the
out-of-fold accuracy/AUC/per-generator breakdown; the weights shipped in
calibration.json are then refit on the full set for the best final ensemble,
but the *reported numbers* come only from data each fold never trained on.

Usage:
    uv run python scripts/evaluate.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from app.calibration import save_calibration
from app.detectors.registry import build_ensemble

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "eval" / "manifest.csv"
N_FOLDS = 5


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def load_manifest() -> list[tuple[Path, int, str]]:
    rows = []
    with open(MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            label = 1 if r["label"] == "ai" else 0
            rows.append((ROOT / r["path"], label, r["generator"]))
    return rows


def compute_model_scores(rows: list[tuple[Path, int, str]]) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Returns (scores [N, M], model_names, labels [N], generators [N])."""
    adapters = build_ensemble()
    for a in adapters:
        a.load()
    model_names = [a.name for a in adapters]

    scores = []
    labels = []
    generators = []
    for i, (path, label, generator) in enumerate(rows):
        img = Image.open(path).convert("RGB")
        row_scores = [a.predict(img) for a in adapters]
        scores.append(row_scores)
        labels.append(label)
        generators.append(generator)
        if (i + 1) % 40 == 0:
            print(f"  scored {i + 1}/{len(rows)}")

    print(f"  scored {len(scores)}/{len(rows)} total")
    return np.array(scores), model_names, np.array(labels), generators


def fit_weights(train_scores: np.ndarray, train_labels: np.ndarray, model_names: list[str]) -> dict[str, float]:
    """AUC-proportional weights: a model at chance (AUC 0.5) gets ~0 weight."""
    weights = {}
    for j, name in enumerate(model_names):
        try:
            auc = roc_auc_score(train_labels, train_scores[:, j])
        except ValueError:
            auc = 0.5
        weights[name] = max(auc - 0.5, 1e-3)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def weighted_logit(scores_row: np.ndarray, model_names: list[str], weights: dict[str, float]) -> float:
    num = sum(weights[name] * _logit(scores_row[j]) for j, name in enumerate(model_names))
    den = sum(weights.values())
    return num / den


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found. Run scripts/fetch_eval_set.py first.")
        sys.exit(1)

    rows = load_manifest()
    print(f"Loaded manifest: {len(rows)} images")

    print("Scoring all images with the ensemble...")
    combined_scores, model_names, labels, generators = compute_model_scores(rows)
    n = len(labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=11997733)
    oof_prob = np.zeros(n)
    oof_raw_logit = np.zeros(n)
    per_model_auc_folds: dict[str, list[float]] = defaultdict(list)

    print(f"\nRunning {N_FOLDS}-fold stratified CV (fit on train fold, predict on held-out fold)...")
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(combined_scores, labels)):
        train_scores, train_labels = combined_scores[train_idx], labels[train_idx]
        test_scores = combined_scores[test_idx]

        weights = fit_weights(train_scores, train_labels, model_names)
        for name, w in weights.items():
            per_model_auc_folds[name].append(w)

        train_raw_logits = np.array([weighted_logit(row, model_names, weights) for row in train_scores])
        test_raw_logits = np.array([weighted_logit(row, model_names, weights) for row in test_scores])

        platt = LogisticRegression()
        platt.fit(train_raw_logits.reshape(-1, 1), train_labels)
        test_prob = platt.predict_proba(test_raw_logits.reshape(-1, 1))[:, 1]

        oof_prob[test_idx] = test_prob
        oof_raw_logit[test_idx] = test_raw_logits
        fold_acc = ((test_prob >= 0.5).astype(int) == labels[test_idx]).mean()
        print(f"  fold {fold_i + 1}: n_test={len(test_idx)}  accuracy={fold_acc:.3f}")

    oof_pred = (oof_prob >= 0.5).astype(int)
    overall_accuracy = float((oof_pred == labels).mean())
    overall_auc = float(roc_auc_score(labels, oof_prob))
    print(f"\nOUT-OF-FOLD accuracy: {overall_accuracy:.3f}   AUC: {overall_auc:.3f}   n={n}")

    per_generator: dict[str, float] = {}
    for gname in sorted(set(generators)):
        idx = [i for i, g in enumerate(generators) if g == gname]
        acc = float((oof_pred[idx] == labels[idx]).mean())
        per_generator[gname] = acc
        print(f"  {gname:20s} n={len(idx):3d}  accuracy={acc:.3f}")

    per_model_auc_final: dict[str, float] = {}
    for j, name in enumerate(model_names):
        try:
            per_model_auc_final[name] = float(roc_auc_score(labels, combined_scores[:, j]))
        except ValueError:
            per_model_auc_final[name] = 0.5
    print("\nPer-model AUC (full dataset, informational only):")
    for name, auc in per_model_auc_final.items():
        print(f"  {name:40s} {auc:.3f}")

    # Refit weights + Platt on the FULL dataset for the shipped ensemble --
    # this is a better final model than any single fold, but its accuracy is
    # NOT what we report (that would be in-sample). Reported numbers above
    # are strictly out-of-fold.
    final_weights = fit_weights(combined_scores, labels, model_names)
    final_raw_logits = np.array([weighted_logit(row, model_names, final_weights) for row in combined_scores])
    final_platt = LogisticRegression()
    final_platt.fit(final_raw_logits.reshape(-1, 1), labels)

    calibration = {
        "ai_ensemble": {
            "weights": final_weights,
            "platt": {"a": float(final_platt.coef_[0][0]), "b": float(final_platt.intercept_[0])},
            "threshold": 0.5,
            "per_model_auc": per_model_auc_final,
            "accuracy": overall_accuracy,
            "auc": overall_auc,
            "n": n,
            "out_of_fold": True,
            "per_generator": per_generator,
        }
    }

    from app.calibration import load_calibration

    existing = load_calibration()
    existing.update(calibration)
    save_calibration(existing)
    print(f"\nWrote calibration to app/calibration.json")


if __name__ == "__main__":
    main()
