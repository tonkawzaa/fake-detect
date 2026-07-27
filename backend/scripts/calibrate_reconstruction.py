"""
Calibrates AEROBLADEDetector (app/detectors/reconstruction.py): asserts
polarity empirically (same discipline as scripts/smoke_test_models.py --
mean reconstruction error on real images must exceed mean error on the
SD-family AI images, not just be assumed from the paper) and fits a 1D
Platt scaling of -distance -> p_ai via stratified K-fold CV, reporting
out-of-fold accuracy/AUC (same discipline as scripts/evaluate.py and
scripts/train_clip_linear_probe.py).

If the polarity assertion fails, this script refuses to write a
calibration entry -- shipping a Platt fit on top of backwards polarity
would produce a confidently wrong secondary signal, worse than no signal
at all, exactly the failure mode scripts/smoke_test_models.py's DROP
outcome exists to prevent for the main ensemble.

Data: data/aeroblade_calib/manifest.csv (scripts/fetch_aeroblade_calib_set.py)
-- small by construction (see that script's docstring: only two SD-VAE-
compatible generator tags exist in the source dataset), so treat the
accuracy number as indicative, not tight. The polarity assertion is the
part that actually gates whether this feature ships.

Usage:
    uv run python scripts/fetch_aeroblade_calib_set.py   # once, if data/aeroblade_calib/ is empty
    uv run python scripts/calibrate_reconstruction.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from app.detectors.reconstruction import get_reconstruction_detector
from app.calibration import load_calibration, save_calibration

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "aeroblade_calib" / "manifest.csv"
N_FOLDS = 5


def load_manifest() -> list[tuple[Path, int, str]]:
    rows = []
    with open(MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            label = 1 if r["label"] == "ai" else 0
            rows.append((ROOT / r["path"], label, r["generator"]))
    return rows


def compute_distances(rows: list[tuple[Path, int, str]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    detector = get_reconstruction_detector()
    detector.load()
    print("Loaded AEROBLADE detector (SD VAE + LPIPS)")

    distances, labels, generators = [], [], []
    for i, (path, label, generator) in enumerate(rows):
        img = Image.open(path).convert("RGB")
        distances.append(detector.reconstruction_error(img))
        labels.append(label)
        generators.append(generator)
        if (i + 1) % 20 == 0:
            print(f"  scored {i + 1}/{len(rows)}")

    return np.array(distances), np.array(labels), generators


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found. Run scripts/fetch_aeroblade_calib_set.py first.")
        sys.exit(1)

    rows = load_manifest()
    n_real = sum(1 for _, label, _ in rows if label == 0)
    n_ai = sum(1 for _, label, _ in rows if label == 1)
    print(f"Loaded manifest: {len(rows)} images ({n_real} real, {n_ai} ai)")
    if n_ai < 8 or n_real < 8:
        print("ERROR: too few images per class to calibrate meaningfully. Widen fetch_aeroblade_calib_set.py's targets.")
        sys.exit(1)

    print("Computing VAE-reconstruction LPIPS distances (this is the slow step: 1 VAE round-trip + LPIPS per image)...")
    distances, labels, generators = compute_distances(rows)

    mean_real = float(distances[labels == 0].mean())
    mean_ai = float(distances[labels == 1].mean())
    polarity_ok = mean_real > mean_ai
    print(f"\nmean reconstruction error on real images: {mean_real:.4f}")
    print(f"mean reconstruction error on AI images   : {mean_ai:.4f}")
    print(f"polarity assertion (mean_real > mean_ai): {'PASS' if polarity_ok else 'FAIL'}")

    if not polarity_ok:
        print(
            "\nPolarity FAILED -- refusing to write a calibration entry. "
            "AEROBLADEDetector.analyze() has no fallback threshold to fall back to (by design -- "
            "see app/detectors/reconstruction.py's module docstring): it will keep returning "
            "p_ai=None / verdict='uncertain' for every image until this is fixed, and "
            "pipeline.py's reconstruction_check will surface only the raw uncalibrated distance."
        )
        sys.exit(1)

    # Feature for the classifier is -distance, so a *higher* value means
    # *more* likely AI, matching every other calibrated score in this repo.
    neg_distance = (-distances).reshape(-1, 1)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=11997733)
    oof_prob = np.zeros(len(labels))

    print(f"\nRunning {N_FOLDS}-fold stratified CV (fit 1D Platt scaling on train fold, predict on held-out fold)...")
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(neg_distance, labels)):
        platt = LogisticRegression()
        platt.fit(neg_distance[train_idx], labels[train_idx])
        oof_prob[test_idx] = platt.predict_proba(neg_distance[test_idx])[:, 1]
        fold_acc = ((oof_prob[test_idx] >= 0.5).astype(int) == labels[test_idx]).mean()
        print(f"  fold {fold_i + 1}: n_test={len(test_idx)}  accuracy={fold_acc:.3f}")

    oof_pred = (oof_prob >= 0.5).astype(int)
    overall_accuracy = float((oof_pred == labels).mean())
    overall_auc = float(roc_auc_score(labels, oof_prob))
    print(f"\nOUT-OF-FOLD accuracy: {overall_accuracy:.3f}   AUC: {overall_auc:.3f}   n={len(labels)}")

    per_generator: dict[str, float] = {}
    for gname in sorted(set(generators)):
        idx = [i for i, g in enumerate(generators) if g == gname]
        acc = float((oof_pred[idx] == labels[idx]).mean())
        per_generator[gname] = acc
        print(f"  {gname:20s} n={len(idx):3d}  accuracy={acc:.3f}")

    # Refit on the full set for the shipped Platt params -- in-sample
    # accuracy deliberately not reported, same discipline as every other
    # calibration script in this repo.
    final_platt = LogisticRegression()
    final_platt.fit(neg_distance, labels)

    # Decision threshold in raw-distance space (informational / for the
    # pipeline's default fallback if it ever needs one): where the fitted
    # Platt curve crosses p=0.5.
    a, b = float(final_platt.coef_[0][0]), float(final_platt.intercept_[0])
    threshold = -b / a if a != 0 else float(np.median(distances))

    calibration = {
        "reconstruction_aeroblade": {
            "platt": {"a": a, "b": b},
            "threshold": threshold,
            "mean_real": mean_real,
            "mean_ai": mean_ai,
            "polarity_ok": True,
            "accuracy": overall_accuracy,
            "auc": overall_auc,
            "n": len(labels),
            "out_of_fold": True,
            "per_generator": per_generator,
            "note": (
                "Only SD-VAE-compatible latent-diffusion generators were used to calibrate this "
                "(see scripts/fetch_aeroblade_calib_set.py); small n, treat accuracy as indicative."
            ),
        }
    }

    existing = load_calibration()
    existing.update(calibration)
    save_calibration(existing)
    print("\nWrote calibration to app/calibration.json (key: reconstruction_aeroblade)")


if __name__ == "__main__":
    main()
