"""
Trains a linear probe on frozen DINOv3 ViT-L/16 features (see
app/detectors/dinov3_features.py for the backbone/feature details), the same
frozen-backbone-plus-linear-head shape as CLIPLinearProbeAdapter
(app/detectors/registry.py). DINOv3LinearProbeAdapter is wired into
build_candidates() and, as of 2026-07-30, ENSEMBLE_MODEL_NAMES -- it
replaced a SigLIP2-giant-based probe that used to occupy this ensemble slot:
DINOv3 ViT-L/16 (~300M params) matched-or-exceeded that probe's measured
accuracy at ~5.8x the inference speed (~1.9B-param vision tower), so the
SigLIP2 probe's adapter and supporting modules were deleted rather than kept
alongside this one. See the ENSEMBLE_MODEL_NAMES comment in registry.py for
the full before/after numbers.

This script only produces dinov3_linear_probe.json -- re-run it and then
scripts/smoke_test_models.py after any change to the feature extraction in
dinov3_features.py (feature = CLS ++ mean-pooled patches, see that module's
docstring for why) before trusting ENSEMBLE_MODEL_NAMES's current weights.

Honest limitation: this script does NOT do perturbation-augmented training
(no JPEG/blur/noise/resize copies) -- it mirrors
train_clip_linear_probe.py's simpler single-clean-copy-per-image shape
instead. Add augmentation later if robustness to those distortions turns
out to matter for this candidate; don't pay for the extra training
complexity up front.

Data: reuses data/clip_train/manifest.csv (scripts/fetch_clip_train_set.py)
-- already disjoint from data/eval/ (evaluate.py's held-out accuracy set).
No new fetch needed; a different model training on the same
already-disjoint set doesn't reopen the leakage risk. Same out-of-fold
discipline as every other calibration script in this repo: the probe is fit
inside stratified K-fold CV and the accuracy/AUC reported is strictly
out-of-fold, then refit on the full set for the shipped weights (in-sample
accuracy of that final refit is deliberately not reported).

Usage:
    uv run python scripts/fetch_clip_train_set.py   # once, if data/clip_train/ is empty
    uv run python scripts/train_dinov3_probe.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from app.detectors.dinov3_features import DINOV3_MODEL_NAME, DINOv3Backbone
from app.detectors.registry import get_device

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "clip_train" / "manifest.csv"
EVAL_MANIFEST_PATH = ROOT / "data" / "eval" / "manifest.csv"
WEIGHTS_PATH = ROOT / "app" / "detectors" / "dinov3_linear_probe.json"
N_FOLDS = 5
SEED = 11997733


def load_manifest(path: Path) -> list[tuple[Path, int, str]]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            label = 1 if r["label"] == "ai" else 0
            rows.append((ROOT / r["path"], label, r["generator"]))
    return rows


def assert_no_overlap_with_eval_set(train_rows: list[tuple[Path, int, str]]) -> None:
    """Defense in depth against training/eval leakage -- see module docstring
    and scripts/fetch_clip_train_set.py. Re-hashes both sets' images (cheap
    at this scale) rather than trusting that the two manifests were built
    correctly, since a silent path mistake here would quietly invalidate
    evaluate.py's accuracy claim."""
    if not EVAL_MANIFEST_PATH.exists():
        return
    eval_rows = load_manifest(EVAL_MANIFEST_PATH)
    eval_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p, _, _ in eval_rows if p.exists()}
    train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p, _, _ in train_rows if p.exists()}
    overlap = eval_hashes & train_hashes
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} image(s) appear in BOTH {MANIFEST_PATH} and {EVAL_MANIFEST_PATH} -- "
            "training the probe on these would leak into evaluate.py's out-of-fold accuracy claim. "
            "Fix data/clip_train/ (re-run scripts/fetch_clip_train_set.py) before proceeding."
        )


def extract_features(rows: list[tuple[Path, int, str]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    backbone = DINOv3Backbone(device=get_device())
    print(f"Loaded {DINOV3_MODEL_NAME} ({backbone.feature_dim}-dim features) on {backbone.device}")

    features, labels, generators = [], [], []
    for i, (path, label, generator) in enumerate(rows):
        img = Image.open(path).convert("RGB")
        feat = backbone.extract(img).cpu().numpy()
        features.append(feat)
        labels.append(label)
        generators.append(generator)
        if (i + 1) % 40 == 0:
            print(f"  extracted {i + 1}/{len(rows)}")

    return np.array(features), np.array(labels), generators


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found. Run scripts/fetch_clip_train_set.py first.")
        sys.exit(1)

    rows = load_manifest(MANIFEST_PATH)
    print(f"Loaded manifest: {len(rows)} images")

    print("Checking data/clip_train/ has zero overlap with data/eval/ (leakage guard)...")
    assert_no_overlap_with_eval_set(rows)
    print("  OK, no overlap")

    print("Extracting frozen DINOv3 ViT-L/16 features (backbone is never updated)...")
    features, labels, generators = extract_features(rows)
    n = len(labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_prob = np.zeros(n)

    print(f"\nRunning {N_FOLDS}-fold stratified CV (fit linear probe on train fold, predict on held-out fold)...")
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(features[train_idx], labels[train_idx])
        oof_prob[test_idx] = clf.predict_proba(features[test_idx])[:, 1]
        fold_acc = ((oof_prob[test_idx] >= 0.5).astype(int) == labels[test_idx]).mean()
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

    # Refit on the full dataset for the shipped probe -- a better final
    # classifier than any single fold, but its in-sample accuracy is
    # deliberately not reported (same discipline as scripts/evaluate.py).
    final_clf = LogisticRegression(max_iter=2000, C=1.0)
    final_clf.fit(features, labels)

    probe = {
        "model_name": DINOV3_MODEL_NAME,
        "feature_dim": int(features.shape[1]),
        "weight": final_clf.coef_[0].tolist(),
        "bias": float(final_clf.intercept_[0]),
        "accuracy": overall_accuracy,
        "auc": overall_auc,
        "n": n,
        "out_of_fold": True,
        "per_generator": per_generator,
    }
    WEIGHTS_PATH.write_text(json.dumps(probe, indent=2))
    print(f"\nWrote linear probe weights to {WEIGHTS_PATH}")
    print(
        "\nNext: run `uv run python scripts/smoke_test_models.py` to check "
        "polarity/accuracy against these newly-trained weights before trusting "
        "dinov3-vit-l16-linear-probe's current place in ENSEMBLE_MODEL_NAMES."
    )


if __name__ == "__main__":
    main()
