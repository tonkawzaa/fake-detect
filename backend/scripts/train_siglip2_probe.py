"""
Trains the linear probe for SigLIP2GiantLinearProbeAdapter
(app/detectors/registry.py), reproducing vincentlc's NTIRE 2026 submission
("Robust AI-Generated Image Detection via SigLIP2-Giant and
Perturbation-Aware Training"): a frozen SigLIP2-giant vision tower
(mean-pooled final-layer patch tokens, see app/detectors/siglip2_features.py)
topped with a single trained linear layer, trained on a mix of clean and
perturbed (JPEG/blur/noise/resize-degraded, see app/detectors/perturbations.py)
copies of each training image so the linear head sees some distribution
shift during training, not just clean images.

Honest limitations vs. the actual submission:
  - The perturbation pipeline is our own approximation, not NTIRE's
    official one (see app/detectors/perturbations.py's docstring) -- any
    robustness this produces is not validated against the actual
    challenge's "robust ROC AUC" metric.
  - vincentlc's writeup doesn't clearly state whether the SigLIP2 backbone
    itself was fine-tuned or frozen; this script always freezes it (a
    literal "linear probe", matching how this repo already does CLIP's
    probe in scripts/train_clip_linear_probe.py) rather than backpropagating
    through a 1.9B-parameter model, which isn't a class of training this
    repo's local-only, no-training-infra setup is built for.

Data: reuses data/clip_train/manifest.csv (scripts/fetch_clip_train_set.py)
-- already disjoint from data/eval/ (evaluate.py's held-out accuracy set),
already verified via content-hash guard when it was built for the CLIP
probe. No new fetch needed; a different model training on the same
already-disjoint images doesn't reintroduce that leakage risk.

K-fold detail that matters here specifically: each source image contributes
multiple feature rows (1 clean + N_AUGMENTED_COPIES perturbed). A plain
StratifiedKFold over the expanded rows would let augmented copies of a
held-out image leak into the training fold (they're correlated with their
clean counterpart, not independent draws) -- this uses
StratifiedGroupKFold, grouping by source image, so all copies of a given
image stay in the same fold together. Reported accuracy is per SOURCE
IMAGE (OOF probability averaged across that image's clean+augmented
copies), not per augmented row, so "n" here means the same thing it means
in every other calibration script in this repo.

Usage:
    uv run python scripts/train_siglip2_probe.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from app.detectors.perturbations import random_perturb
from app.detectors.registry import get_device
from app.detectors.siglip2_features import SIGLIP2_REPO_ID, SigLIP2Backbone

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "clip_train" / "manifest.csv"
WEIGHTS_PATH = ROOT / "app" / "detectors" / "siglip2_giant_linear_probe.json"
N_FOLDS = 5
N_AUGMENTED_COPIES = 2  # per image, in addition to the clean copy
SEED = 11997733


def load_manifest() -> list[tuple[Path, int, str]]:
    rows = []
    with open(MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            label = 1 if r["label"] == "ai" else 0
            rows.append((ROOT / r["path"], label, r["generator"]))
    return rows


def extract_features(
    rows: list[tuple[Path, int, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns (features [n*(1+N_AUGMENTED_COPIES), dim], labels, group_ids
    [same source image -> same id], generators) -- expanded rows, one per
    clean/augmented copy."""
    backbone = SigLIP2Backbone(device=get_device())
    print(f"Loaded {SIGLIP2_REPO_ID} ({backbone.feature_dim}-dim features) on {backbone.device}")
    rng = random.Random(SEED)

    features, labels, groups, generators = [], [], [], []
    for i, (path, label, generator) in enumerate(rows):
        img = Image.open(path).convert("RGB")
        variants = [img] + [random_perturb(img, rng) for _ in range(N_AUGMENTED_COPIES)]
        for variant in variants:
            features.append(backbone.extract(variant).cpu().numpy())
            labels.append(label)
            groups.append(i)
            generators.append(generator)
        if (i + 1) % 20 == 0:
            print(f"  extracted {i + 1}/{len(rows)} images ({len(features)} clean+augmented feature rows so far)")

    return np.array(features), np.array(labels), np.array(groups), generators


def per_image_oof(oof_prob: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Averages OOF probability across each source image's clean+augmented
    copies -- see module docstring for why this, not per-row accuracy, is
    the number that should be reported and compared to other scripts."""
    n_images = groups.max() + 1
    image_prob = np.zeros(n_images)
    image_label = np.zeros(n_images, dtype=int)
    for g in range(n_images):
        mask = groups == g
        image_prob[g] = oof_prob[mask].mean()
        image_label[g] = labels[mask][0]
    return image_prob, image_label


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found. Run scripts/fetch_clip_train_set.py first.")
        sys.exit(1)

    rows = load_manifest()
    print(f"Loaded manifest: {len(rows)} images (each will yield 1 clean + {N_AUGMENTED_COPIES} perturbed feature rows)")

    print("Extracting frozen SigLIP2-giant features (clean + perturbed copies; backbone is never updated)...")
    features, labels, groups, generators = extract_features(rows)
    print(f"Total feature rows: {len(labels)}")

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_prob = np.zeros(len(labels))

    print(f"\nRunning {N_FOLDS}-fold stratified group CV (grouped by source image, fit on train fold, predict on held-out fold)...")
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(features, labels, groups)):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(features[train_idx], labels[train_idx])
        oof_prob[test_idx] = clf.predict_proba(features[test_idx])[:, 1]
        fold_acc = ((oof_prob[test_idx] >= 0.5).astype(int) == labels[test_idx]).mean()
        print(f"  fold {fold_i + 1}: n_rows_test={len(test_idx)}  row_accuracy={fold_acc:.3f}")

    image_prob, image_label = per_image_oof(oof_prob, labels, groups)
    image_pred = (image_prob >= 0.5).astype(int)
    overall_accuracy = float((image_pred == image_label).mean())
    overall_auc = float(roc_auc_score(image_label, image_prob))
    print(f"\nOUT-OF-FOLD accuracy (per source image): {overall_accuracy:.3f}   AUC: {overall_auc:.3f}   n={len(image_label)}")

    # generator per source image (first occurrence == every occurrence, all
    # copies of an image share its generator)
    image_generator = [None] * (groups.max() + 1)
    for g, gen in zip(groups, generators):
        image_generator[g] = gen

    per_generator: dict[str, float] = {}
    for gname in sorted(set(image_generator)):
        idx = [i for i, g in enumerate(image_generator) if g == gname]
        acc = float((image_pred[idx] == image_label[idx]).mean())
        per_generator[gname] = acc
        print(f"  {gname:20s} n={len(idx):3d}  accuracy={acc:.3f}")

    # Refit on the full (clean + augmented) set for the shipped probe --
    # in-sample accuracy deliberately not reported, same discipline as
    # every other calibration script in this repo.
    final_clf = LogisticRegression(max_iter=2000, C=1.0)
    final_clf.fit(features, labels)

    probe = {
        "repo_id": SIGLIP2_REPO_ID,
        "feature_dim": int(features.shape[1]),
        "weight": final_clf.coef_[0].tolist(),
        "bias": float(final_clf.intercept_[0]),
        "accuracy": overall_accuracy,
        "auc": overall_auc,
        "n": len(image_label),
        "n_augmented_copies": N_AUGMENTED_COPIES,
        "out_of_fold": True,
        "per_generator": per_generator,
    }
    WEIGHTS_PATH.write_text(json.dumps(probe, indent=2))
    print(f"\nWrote linear probe weights to {WEIGHTS_PATH}")
    print(
        "\nNext: uv run python scripts/smoke_test_models.py "
        "to check polarity/accuracy on the smoke set before adding "
        "'siglip2-giant-linear-probe' to ENSEMBLE_MODEL_NAMES in registry.py."
    )


if __name__ == "__main__":
    main()
