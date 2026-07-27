"""
Phase 3: beauty-mode sub-features F1-F8.

Every function returns a *raw* float where, roughly, higher = more evidence
of skin smoothing/retouching (except F5, which is a 0..1 guard multiplier,
and F7 which returns a dict of named z-scores). Raw values are unbounded and
get normalized against calibrated thresholds in score.py -- calibration
comes from scripts/calibrate_beauty.py (Phase 5) using real beauty-mode
on/off photo pairs; until that file exists, score.py falls back to
hand-picked defaults documented there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .regions import FaceRegions

EPS = 1e-6


def _lap_var(gray: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 16:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap[mask].var())


def f1_skin_hf_ratio(gray: np.ndarray, regions: FaceRegions) -> float:
    """Reference (eye+border) high-frequency energy vs skin high-frequency energy.
    Beauty filters smooth skin but leave eyes/background untouched, so a
    high ratio (reference sharp, skin not) is evidence of filtering."""
    ref_mask = regions.eye_mask | regions.border_mask
    skin_var = _lap_var(gray, regions.skin_mask)
    ref_var = _lap_var(gray, ref_mask)
    return ref_var / (skin_var + EPS)


def f2_noise_residual_mismatch(gray: np.ndarray, regions: FaceRegions) -> float:
    """Sensor/JPEG noise std in skin vs reference. Smoothing filters strip
    high-frequency sensor noise selectively from skin."""
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=50, sigmaSpace=50)
    residual = cv2.absdiff(gray, denoised).astype(np.float64)
    ref_mask = regions.eye_mask | regions.border_mask
    if regions.skin_mask.sum() < 16 or ref_mask.sum() < 16:
        return 0.0
    skin_std = residual[regions.skin_mask].std()
    ref_std = residual[ref_mask].std()
    return ref_std / (skin_std + EPS)


def _lbp_entropy(gray: np.ndarray, mask: np.ndarray) -> float:
    from skimage.feature import local_binary_pattern

    if mask.sum() < 64:
        return 0.0
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    vals = lbp[mask]
    hist, _ = np.histogram(vals, bins=10, range=(0, 10), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def f3_lbp_entropy_drop(gray: np.ndarray, regions: FaceRegions) -> float:
    """Uniform-LBP entropy: pore/skin-texture detail collapses under smoothing.
    Raw = how much lower skin entropy is than the reference region's."""
    ref_mask = regions.border_mask
    skin_h = _lbp_entropy(gray, regions.skin_mask)
    ref_h = _lbp_entropy(gray, ref_mask)
    if ref_h < EPS:
        return 0.0
    return max(0.0, (ref_h - skin_h) / ref_h)


def _radial_highfreq_fraction(gray_patch: np.ndarray, cutoff_frac: float = 0.35) -> float:
    if gray_patch.size == 0:
        return 0.0
    h, w = gray_patch.shape
    f = np.fft.fftshift(np.fft.fft2(gray_patch.astype(np.float64)))
    power = np.abs(f) ** 2
    cy, cx = h / 2, w / 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = min(cy, cx)
    total = power.sum() + EPS
    high = power[r > cutoff_frac * r_max].sum()
    return float(high / total)


def f4_fft_highfreq_falloff(gray: np.ndarray, regions: FaceRegions) -> float:
    """Fraction of spectral energy in the high-frequency band, skin patch vs
    a background/hair reference patch. Smoothing disproportionately removes
    high-frequency (pore/texture) energy."""
    x, y, w, h = regions.face_bbox
    skin_patch = gray[y : y + h, x : x + w]
    ref = gray.copy()
    ref[~regions.border_mask] = 0
    ys, xs = np.where(regions.border_mask)
    if len(xs) < 32 or skin_patch.size < 32:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ref_patch = gray[y0:y1, x0:x1]
    if ref_patch.size < 32:
        return 0.0
    skin_hf = _radial_highfreq_fraction(skin_patch)
    ref_hf = _radial_highfreq_fraction(ref_patch)
    return max(0.0, ref_hf - skin_hf)


def f5_blur_locality_guard(gray: np.ndarray, regions: FaceRegions) -> float:
    """Guard multiplier in [0,1]. If the *background/reference* region is
    itself soft (out-of-focus photo, heavy compression, low resolution),
    the skin-vs-reference contrast features are unreliable evidence of
    filtering -- this damps the fused beauty score rather than letting a
    globally blurry photo read as 'filtered'."""
    ref_var = _lap_var(gray, regions.border_mask | regions.eye_mask)
    # calibrated in Phase 5; conservative default midpoint for a 256px-scale image
    sharp_floor, sharp_ceiling = 8.0, 60.0
    if ref_var <= sharp_floor:
        return 0.15
    if ref_var >= sharp_ceiling:
        return 1.0
    return 0.15 + 0.85 * (ref_var - sharp_floor) / (sharp_ceiling - sharp_floor)


def f6_skin_tone_uniformity(image_rgb: np.ndarray, regions: FaceRegions) -> float:
    """Even-tone / whitening signal: local lightness variance in the skin
    region (Lab L channel) is suppressed by smoothing/whitening filters."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float64)
    if regions.skin_mask.sum() < 16 or regions.border_mask.sum() < 16:
        return 0.0
    skin_std = L[regions.skin_mask].std()
    ref_std = L[regions.border_mask].std()
    if ref_std < EPS:
        return 0.0
    return max(0.0, (ref_std - skin_std) / ref_std)


# Rough population priors (NOT fit on our eval set yet -- calibrate_beauty.py
# overwrites these with (mean, std) measured on data/eval/real once Phase 5
# runs). Wide std means this feature stays near-inert until calibrated.
_F7_PRIORS: dict[str, tuple[float, float]] = {
    "eye_aspect_ratio": (0.30, 0.08),
    "jaw_to_cheekbone_ratio": (0.90, 0.12),
    "nose_width_to_face_ratio": (0.24, 0.06),
    "chin_length_to_face_ratio": (0.18, 0.06),
}


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def f7_geometry_zscores(
    landmarks_norm: list[tuple[float, float, float]],
    width: int,
    height: int,
    priors: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Face-shape ratios compared to a baseline population (data/eval/real).
    NOT a filter detector on its own -- an unusual ratio may just mean a face
    that differs from the baseline population, not retouching. Report this
    with the population it's compared against, never as a bare anomaly."""
    priors = priors or _F7_PRIORS
    pts = [(p[0] * width, p[1] * height) for p in landmarks_norm]

    def P(i: int) -> tuple[float, float]:
        return pts[i]

    # MediaPipe FaceMesh canonical indices (stable across the 468/478 topology).
    left_eye_outer, left_eye_inner = P(33), P(133)
    right_eye_inner, right_eye_outer = P(362), P(263)
    left_eye_top, left_eye_bottom = P(159), P(145)
    jaw_left, jaw_right = P(58), P(288)
    cheek_left, cheek_right = P(234), P(454)
    nose_left, nose_right = P(48), P(278)
    chin, nose_base = P(152), P(2)
    face_top, face_bottom = P(10), P(152)

    interocular = _dist(left_eye_inner, right_eye_inner) + EPS
    eye_aspect_ratio = _dist(left_eye_top, left_eye_bottom) / (_dist(left_eye_outer, left_eye_inner) + EPS)
    face_width = _dist(cheek_left, cheek_right) + EPS
    face_height = _dist(face_top, face_bottom) + EPS
    jaw_to_cheekbone_ratio = _dist(jaw_left, jaw_right) / face_width
    nose_width_to_face_ratio = _dist(nose_left, nose_right) / face_width
    chin_length_to_face_ratio = _dist(nose_base, chin) / face_height

    raw = {
        "eye_aspect_ratio": eye_aspect_ratio,
        "jaw_to_cheekbone_ratio": jaw_to_cheekbone_ratio,
        "nose_width_to_face_ratio": nose_width_to_face_ratio,
        "chin_length_to_face_ratio": chin_length_to_face_ratio,
    }
    zscores = {}
    for key, value in raw.items():
        mean, std = priors.get(key, (value, 1.0))
        zscores[key] = (value - mean) / (std + EPS)
    return zscores


def f8_contour_warp(gray: np.ndarray, regions: FaceRegions) -> float:
    """Best-effort, low-confidence signal: slimming/liquify warps bend
    straight background lines near the jaw contour. We compare Hough-detected
    line-segment length near the jawline band vs. elsewhere in the border
    region. A weak, noisy feature by design -- weighted low in fusion."""
    x, y, w, h = regions.face_bbox
    jaw_band = np.zeros_like(gray, dtype=bool)
    band_px = max(4, int(0.06 * h))
    y0, y1 = min(gray.shape[0], y + h), min(gray.shape[0], y + h + band_px)
    x0, x1 = max(0, x), min(gray.shape[1], x + w)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    jaw_band[y0:y1, x0:x1] = True
    jaw_band &= ~regions.skin_mask

    def mean_line_length(mask: np.ndarray) -> float:
        if mask.sum() < 64:
            return -1.0
        region = gray.copy()
        region[~mask] = 0
        edges = cv2.Canny(region, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=8, maxLineGap=3)
        if lines is None:
            return 0.0
        lengths = [math.hypot(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in lines[:, 0]]
        return float(np.mean(lengths)) if lengths else 0.0

    jaw_len = mean_line_length(jaw_band)
    ref_len = mean_line_length(regions.border_mask)
    if jaw_len < 0 or ref_len <= EPS:
        return 0.0
    return max(0.0, (ref_len - jaw_len) / ref_len)


@dataclass
class RawFeatures:
    f1_skin_hf_ratio: float
    f2_noise_residual: float
    f3_lbp_entropy_drop: float
    f4_fft_highfreq: float
    f5_guard: float
    f6_tone_uniformity: float
    f7_geometry_z: dict[str, float]
    f8_contour_warp: float


def compute_all_features(image_rgb: np.ndarray, landmarks_norm: list[tuple[float, float, float]], regions: FaceRegions) -> RawFeatures:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    height, width = image_rgb.shape[:2]
    return RawFeatures(
        f1_skin_hf_ratio=f1_skin_hf_ratio(gray, regions),
        f2_noise_residual=f2_noise_residual_mismatch(gray, regions),
        f3_lbp_entropy_drop=f3_lbp_entropy_drop(gray, regions),
        f4_fft_highfreq=f4_fft_highfreq_falloff(gray, regions),
        f5_guard=f5_blur_locality_guard(gray, regions),
        f6_tone_uniformity=f6_skin_tone_uniformity(image_rgb, regions),
        f7_geometry_z=f7_geometry_zscores(landmarks_norm, width, height),
        f8_contour_warp=f8_contour_warp(gray, regions),
    )
