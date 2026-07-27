"""
Phase 5: calibrate F1-F8 thresholds from paired beauty-filter on/off photos.

Preferred input: data/beauty_pairs/on/*.jpg and data/beauty_pairs/off/*.jpg,
photos of the same faces shot with a phone's beauty mode on vs. off. This is
real, labelled evidence of what a filter actually does to the pixel
statistics -- nothing else substitutes for it, per the plan.

Bootstrap fallback (used automatically when no real pairs are present, and
clearly labelled as such in the output and in calibration.json): synthesizes
an "on" set by applying a skin-only smoothing pass (bilateral filter + mild
lightness lift, restricted to the FaceMesh skin mask) to the real photos in
data/eval/real/. This is NOT a real beauty app and will not capture what
every filter does, but it exercises the same physical effect (skin smoothed,
eyes/background untouched) the features are designed to detect, which is
enough to fit sane midpoints instead of the current hand-picked guesses.

Usage:
    uv run python scripts/calibrate_beauty.py
    # to force the bootstrap even if data/beauty_pairs has content:
    uv run python scripts/calibrate_beauty.py --bootstrap
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image

from app.beauty.features import RawFeatures, compute_all_features
from app.beauty.regions import FaceRegions, build_face_regions
from app.beauty.score import DEFAULT_THRESHOLDS
from app.calibration import load_calibration, save_calibration
from app.detectors.face_gate import get_face_gate

ROOT = Path(__file__).resolve().parents[1]
PAIRS_ON = ROOT / "data" / "beauty_pairs" / "on"
PAIRS_OFF = ROOT / "data" / "beauty_pairs" / "off"
BOOTSTRAP_SOURCE = ROOT / "data" / "eval" / "real"
MIN_REAL_PAIRS = 10

FEATURE_KEYS = ["f1_skin_hf_ratio", "f2_noise_residual", "f3_lbp_entropy_drop", "f4_fft_highfreq", "f6_tone_uniformity", "f8_contour_warp"]


def _apply_synthetic_skin_smoothing(image: Image.Image, regions: FaceRegions) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    smoothed = cv2.bilateralFilter(rgb, d=15, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.06 + 4, 0, 255)  # mild brighten/whiten
    smoothed = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    mask3 = np.repeat(regions.skin_mask[:, :, None], 3, axis=2)
    out = np.where(mask3, smoothed, rgb)
    return Image.fromarray(out)


def collect_features(image_paths: list[Path]) -> list[RawFeatures]:
    gate = get_face_gate()
    out = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        g = gate.run(img)
        if g.status != "ok":
            continue
        regions = build_face_regions(g.face.landmarks_norm, img.width, img.height)
        rgb = np.array(img)
        out.append(compute_all_features(rgb, g.face.landmarks_norm, regions))
    return out


def collect_bootstrap_pairs(max_n: int = 80) -> tuple[list[RawFeatures], list[RawFeatures]]:
    gate = get_face_gate()
    off_feats, on_feats = [], []
    paths = sorted(BOOTSTRAP_SOURCE.glob("*.jpg"))[:max_n]
    for p in paths:
        img = Image.open(p).convert("RGB")
        g = gate.run(img)
        if g.status != "ok":
            continue
        regions = build_face_regions(g.face.landmarks_norm, img.width, img.height)
        rgb = np.array(img)
        off_feats.append(compute_all_features(rgb, g.face.landmarks_norm, regions))

        synthetic_on = _apply_synthetic_skin_smoothing(img, regions)
        g2 = gate.run(synthetic_on)  # landmarks barely change, but stay consistent
        regions2 = build_face_regions(g2.face.landmarks_norm, synthetic_on.width, synthetic_on.height)
        on_feats.append(compute_all_features(np.array(synthetic_on), g2.face.landmarks_norm, regions2))
    return off_feats, on_feats


def fit_midpoint_scale(off_values: np.ndarray, on_values: np.ndarray) -> tuple[float, float]:
    """Midpoint halfway between the two class means (in log-space if all
    positive, to keep skewed ratio features well-behaved); scale from the
    pooled spread so ~1 pooled-std separates 0.5 from ~0.73 on the squash."""
    midpoint = float((np.median(off_values) + np.median(on_values)) / 2.0)
    pooled_std = float(np.concatenate([off_values, on_values]).std())
    scale = max(pooled_std * 0.5, 1e-4)
    return midpoint, scale


def main() -> None:
    use_bootstrap = "--bootstrap" in sys.argv
    n_on = len(list(PAIRS_ON.glob("*"))) if PAIRS_ON.exists() else 0
    n_off = len(list(PAIRS_OFF.glob("*"))) if PAIRS_OFF.exists() else 0

    if not use_bootstrap and n_on >= MIN_REAL_PAIRS and n_off >= MIN_REAL_PAIRS:
        print(f"Using REAL beauty on/off pairs: {n_on} on, {n_off} off")
        on_feats = collect_features(sorted(PAIRS_ON.glob("*")))
        off_feats = collect_features(sorted(PAIRS_OFF.glob("*")))
        source = "real_pairs"
    else:
        print(
            f"No real beauty pairs found ({n_on} on / {n_off} off, need >= {MIN_REAL_PAIRS} each) "
            f"-- falling back to SYNTHETIC bootstrap pairs from {BOOTSTRAP_SOURCE}.\n"
            "This is clearly marked in calibration.json as source='synthetic_bootstrap'. "
            "Drop real phone photos (beauty mode on vs off, same faces) into "
            "data/beauty_pairs/{on,off}/ and re-run for a properly calibrated score."
        )
        off_feats, on_feats = collect_bootstrap_pairs()
        source = "synthetic_bootstrap"

    print(f"Collected {len(off_feats)} off-samples, {len(on_feats)} on-samples")
    if len(off_feats) < 5 or len(on_feats) < 5:
        print("ERROR: too few usable samples (face gate rejected most images). Aborting.")
        sys.exit(1)

    thresholds: dict[str, list[float]] = {}
    for key in FEATURE_KEYS:
        off_vals = np.array([getattr(f, key) for f in off_feats])
        on_vals = np.array([getattr(f, key) for f in on_feats])
        midpoint, scale = fit_midpoint_scale(off_vals, on_vals)
        thresholds[key] = [midpoint, scale]
        print(f"  {key:22s} off_median={np.median(off_vals):.4f}  on_median={np.median(on_vals):.4f}  -> midpoint={midpoint:.4f} scale={scale:.4f}")

    separation_note = None
    try:
        from app.beauty.score import score_beauty

        off_scores = [score_beauty(f, thresholds={k: tuple(v) for k, v in thresholds.items()}).score for f in off_feats]
        on_scores = [score_beauty(f, thresholds={k: tuple(v) for k, v in thresholds.items()}).score for f in on_feats]
        print(f"\nFused score check: off mean={np.mean(off_scores):.3f}  on mean={np.mean(on_scores):.3f}")
        separation_note = f"off mean={np.mean(off_scores):.3f}, on mean={np.mean(on_scores):.3f}"
        if np.mean(on_scores) <= np.mean(off_scores):
            print("WARNING: 'on' (filtered) mean is not higher than 'off' -- calibration did not separate the classes.")
    except Exception as e:
        print(f"WARN: could not run separation check: {e}")

    calibration = load_calibration()
    calibration["beauty"] = {
        "thresholds": thresholds,
        "source": source,
        "n_on": len(on_feats),
        "n_off": len(off_feats),
        "separation_check": separation_note,
    }
    save_calibration(calibration)
    print("\nWrote beauty calibration to app/calibration.json")


if __name__ == "__main__":
    main()
