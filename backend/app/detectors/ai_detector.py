"""
Phase 2: AI-generation detection.

Runs the ensemble (built in registry.build_ensemble(), membership decided by
the Phase 0 smoke test) on two crops per image -- the full frame and a tight
face crop -- because AI face artifacts concentrate in the face while global
cues (lighting consistency, background) show up in the full frame.

Fusion is logit-space averaging with per-model weights, then Platt/temperature
calibration fit in scripts/evaluate.py (Phase 5) and loaded from
calibration.json at startup. Until that calibration file exists, weights
default to equal and the sigmoid is uncalibrated (raw ensemble average) --
main.py surfaces this via /model-info so the UI doesn't imply a calibrated
number it doesn't have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .registry import DetectorAdapter, build_ensemble

FACE_CROP_MARGIN = 0.20  # fraction of bbox size added on each side


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def crop_face(image: Image.Image, bbox_px: tuple[int, int, int, int], margin: float = FACE_CROP_MARGIN) -> Image.Image:
    x, y, w, h = bbox_px
    mx, my = int(w * margin), int(h * margin)
    left = max(0, x - mx)
    top = max(0, y - my)
    right = min(image.width, x + w + mx)
    bottom = min(image.height, y + h + my)
    return image.crop((left, top, right, bottom))


@dataclass
class ModelScore:
    name: str
    p_ai_full: float
    p_ai_face: float
    p_ai_combined: float  # mean of full+face for this model
    weight: float
    eval_auc: float | None = None


@dataclass
class EnsembleResult:
    ai_probability: float  # calibrated, 0..1
    raw_ensemble_logit: float
    verdict: str  # "likely_ai" | "likely_real" | "uncertain"
    confidence_band: str  # "high" | "medium" | "low"
    models: list[ModelScore] = field(default_factory=list)
    calibrated: bool = False


class AIDetectorEnsemble:
    def __init__(self):
        self._adapters: list[DetectorAdapter] | None = None
        self.calibration: dict | None = None  # set via set_calibration()

    def _ensure_loaded(self) -> list[DetectorAdapter]:
        if self._adapters is None:
            adapters = build_ensemble()
            for a in adapters:
                a.load()
            self._adapters = adapters
        return self._adapters

    def set_calibration(self, calibration: dict | None) -> None:
        """calibration = {"weights": {name: w}, "platt": {"a": .., "b": ..},
        "threshold": 0.5, "per_model_auc": {name: auc}}"""
        self.calibration = calibration

    def analyze(self, full_image: Image.Image, bbox_px: tuple[int, int, int, int]) -> EnsembleResult:
        adapters = self._ensure_loaded()
        face_crop = crop_face(full_image, bbox_px)

        weights_cfg = (self.calibration or {}).get("weights", {})
        aucs_cfg = (self.calibration or {}).get("per_model_auc", {})
        default_weight = 1.0 / max(len(adapters), 1)

        model_scores: list[ModelScore] = []
        weighted_logit_sum = 0.0
        weight_sum = 0.0

        for adapter in adapters:
            p_full = adapter.predict(full_image)
            p_face = adapter.predict(face_crop)
            p_combined = (p_full + p_face) / 2.0
            weight = float(weights_cfg.get(adapter.name, default_weight))

            weighted_logit_sum += weight * _logit(p_combined)
            weight_sum += weight

            model_scores.append(
                ModelScore(
                    name=adapter.name,
                    p_ai_full=p_full,
                    p_ai_face=p_face,
                    p_ai_combined=p_combined,
                    weight=weight,
                    eval_auc=aucs_cfg.get(adapter.name),
                )
            )

        raw_logit = weighted_logit_sum / weight_sum if weight_sum > 0 else 0.0

        platt = (self.calibration or {}).get("platt")
        calibrated = platt is not None
        if calibrated:
            final_logit = platt["a"] * raw_logit + platt["b"]
        else:
            final_logit = raw_logit
        ai_probability = _sigmoid(final_logit)

        threshold = (self.calibration or {}).get("threshold", 0.5)
        # agreement: how close model scores are to each other (low std = high agreement)
        combined_scores = np.array([m.p_ai_combined for m in model_scores])
        agreement = 1.0 - min(float(np.std(combined_scores)) * 2.0, 1.0) if len(combined_scores) > 1 else 1.0
        margin = abs(ai_probability - threshold)

        if margin < 0.10 or agreement < 0.5:
            verdict = "uncertain"
        elif ai_probability >= threshold:
            verdict = "likely_ai"
        else:
            verdict = "likely_real"

        if agreement >= 0.75 and margin >= 0.25:
            band = "high"
        elif agreement >= 0.5 and margin >= 0.10:
            band = "medium"
        else:
            band = "low"

        return EnsembleResult(
            ai_probability=ai_probability,
            raw_ensemble_logit=raw_logit,
            verdict=verdict,
            confidence_band=band,
            models=model_scores,
            calibrated=calibrated,
        )


_ensemble_singleton: AIDetectorEnsemble | None = None


def get_ensemble() -> AIDetectorEnsemble:
    global _ensemble_singleton
    if _ensemble_singleton is None:
        _ensemble_singleton = AIDetectorEnsemble()
    return _ensemble_singleton
