"""
Phase 3: fuse F1-F8 into a transparent weighted beauty-mode score.

No trained classifier here -- we don't have beauty on/off labels at scale
(this is exactly what scripts/calibrate_beauty.py, Phase 5, exists to
partially fix by fitting the per-feature normalization midpoints from
paired on/off phone photos). Until that calibration file is produced, the
DEFAULT_THRESHOLDS below are hand-picked from the smoke-set feature ranges
and should be treated as a starting point, not a validated cutoff.

Every sub-score is surfaced in the report, not just the fused number --
that's what keeps this debuggable rather than a black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .features import RawFeatures

# (midpoint, scale) for a logistic squashing of each raw feature into 0..1.
# raw=midpoint -> 0.5; raw=midpoint+scale -> ~0.73; overwritten by
# scripts/calibrate_beauty.py once beauty on/off pairs are collected.
DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "f1_skin_hf_ratio": (1.8, 1.0),
    "f2_noise_residual": (1.3, 0.6),
    "f3_lbp_entropy_drop": (0.12, 0.10),
    "f4_fft_highfreq": (0.01, 0.01),
    "f6_tone_uniformity": (0.20, 0.15),
    "f8_contour_warp": (0.20, 0.20),
}

# Weights per the plan: F1/F2/F3 primary/high, F4/F6 medium, F7/F8 low.
WEIGHTS: dict[str, float] = {
    "f1_skin_hf_ratio": 1.0,
    "f2_noise_residual": 0.8,
    "f3_lbp_entropy_drop": 0.8,
    "f4_fft_highfreq": 0.5,
    "f6_tone_uniformity": 0.5,
    "f7_geometry": 0.25,
    "f8_contour_warp": 0.25,
}

LEVEL_BUCKETS = [
    (0.30, "None"),
    (0.50, "Light"),
    (0.70, "Moderate"),
    (1.01, "Heavy"),
]


def _squash(raw: float, midpoint: float, scale: float) -> float:
    import math

    if scale <= 0:
        scale = 1e-3
    return 1.0 / (1.0 + math.exp(-(raw - midpoint) / scale))


def _f7_subscore(zscores: dict[str, float]) -> float:
    # mean absolute z-score, squashed; low weight and always annotated as a
    # population comparison rather than a standalone filter detector.
    if not zscores:
        return 0.0
    mean_abs_z = sum(abs(v) for v in zscores.values()) / len(zscores)
    return _squash(mean_abs_z, midpoint=1.5, scale=1.0)


@dataclass
class BeautyReport:
    score: float  # 0..1, after the F5 guard
    level: str
    subscores: dict[str, float] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)
    guard_multiplier: float = 1.0
    notes: list[str] = field(default_factory=list)


def score_beauty(features: RawFeatures, thresholds: dict[str, tuple[float, float]] | None = None) -> BeautyReport:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    subscores: dict[str, float] = {
        "f1_skin_hf_ratio": _squash(features.f1_skin_hf_ratio, *thresholds["f1_skin_hf_ratio"]),
        "f2_noise_residual": _squash(features.f2_noise_residual, *thresholds["f2_noise_residual"]),
        "f3_lbp_entropy_drop": _squash(features.f3_lbp_entropy_drop, *thresholds["f3_lbp_entropy_drop"]),
        "f4_fft_highfreq": _squash(features.f4_fft_highfreq, *thresholds["f4_fft_highfreq"]),
        "f6_tone_uniformity": _squash(features.f6_tone_uniformity, *thresholds["f6_tone_uniformity"]),
        "f7_geometry": _f7_subscore(features.f7_geometry_z),
        "f8_contour_warp": _squash(features.f8_contour_warp, *thresholds["f8_contour_warp"]),
    }

    weight_sum = sum(WEIGHTS.values())
    fused = sum(subscores[k] * WEIGHTS[k] for k in subscores) / weight_sum

    guarded = fused * features.f5_guard
    notes: list[str] = []
    if features.f5_guard < 0.5:
        notes.append(
            "Background/eye region is itself soft (low resolution, out-of-focus, or heavily "
            "compressed) -- skin-smoothing evidence is unreliable here, so the score has been damped."
        )
    if any(abs(v) > 2.0 for v in features.f7_geometry_z.values()):
        notes.append(
            "Facial proportions differ notably from the baseline photo population used for "
            "comparison -- this can reflect natural variation, not necessarily retouching."
        )

    level = next(name for cutoff, name in LEVEL_BUCKETS if guarded < cutoff)

    return BeautyReport(
        score=guarded,
        level=level,
        subscores=subscores,
        raw={
            "f1_skin_hf_ratio": features.f1_skin_hf_ratio,
            "f2_noise_residual": features.f2_noise_residual,
            "f3_lbp_entropy_drop": features.f3_lbp_entropy_drop,
            "f4_fft_highfreq": features.f4_fft_highfreq,
            "f5_guard": features.f5_guard,
            "f6_tone_uniformity": features.f6_tone_uniformity,
            "f8_contour_warp": features.f8_contour_warp,
            **{f"f7_{k}_z": v for k, v in features.f7_geometry_z.items()},
        },
        guard_multiplier=features.f5_guard,
        notes=notes,
    )
