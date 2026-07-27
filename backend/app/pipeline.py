"""
Orchestrates one /analyze request: face gate -> AI ensemble + beauty engine
+ provenance -> fused report. Each analysis stage is independent (a failure
in provenance parsing shouldn't take down the AI verdict), so stages are
wrapped defensively and degrade to "unavailable" rather than 500ing the
whole request.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

from .beauty.features import compute_all_features
from .beauty.regions import build_face_regions
from .beauty.score import score_beauty
from .calibration import load_calibration
from .detectors.ai_detector import get_ensemble
from .detectors.face_gate import get_face_gate
from .detectors.heatmap import compute_saliency_map, render_overlay_png_base64
from .detectors.reconstruction import get_reconstruction_detector
from .detectors.registry import build_ensemble
from .forensics.credentials import read_c2pa
from .forensics.metadata import read_exif
from .schemas import (
    AnalyzeReport,
    BeautyOut,
    FaceOut,
    FaceQualityOut,
    ModelAccuracyOut,
    ModelScoreOut,
    ProvenanceOut,
    ReconstructionCheckOut,
)

logger = logging.getLogger("pipeline")

LIMITATIONS = [
    "This is a triage tool, not proof: independent 2026 benchmarks put the best "
    "zero-shot open detector around 75% mean accuracy across generators.",
    "Detection accuracy drops sharply on the newest commercial generators "
    "(e.g. Flux Dev, Midjourney v7, Firefly v4) -- as low as 18-30% in published benchmarks.",
    "The beauty-mode score is a transparent heuristic over pixel statistics, not a "
    "trained classifier -- treat it as an indicator to inspect, not a verdict.",
]


def _model_accuracy_out(calibration: dict) -> ModelAccuracyOut:
    ai_cal = calibration.get("ai_ensemble") or {}
    if "accuracy" not in ai_cal:
        return ModelAccuracyOut(note="Not yet calibrated -- run scripts/evaluate.py to measure this on a held-out eval set.")
    return ModelAccuracyOut(
        accuracy=ai_cal.get("accuracy"),
        auc=ai_cal.get("auc"),
        n=ai_cal.get("n"),
        out_of_fold=ai_cal.get("out_of_fold", False),
        per_generator=ai_cal.get("per_generator"),
    )


def analyze_image(image_bytes: bytes) -> AnalyzeReport:
    calibration = load_calibration()
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image.load()

    gate = get_face_gate()
    gate_result = gate.run(pil_image)

    face_out = FaceOut(
        count=gate_result.face_count,
        bbox=gate_result.face.bbox_px if gate_result.face else None,
        quality=(
            FaceQualityOut(
                width=gate_result.quality.width,
                height=gate_result.quality.height,
                blur_score=gate_result.quality.blur_score,
                blur_label=gate_result.quality.blur_label,
                coverage=gate_result.quality.coverage,
            )
            if gate_result.quality
            else None
        ),
        notes=gate_result.notes,
    )

    if gate_result.status != "ok":
        return AnalyzeReport(
            status=gate_result.status,
            message=gate_result.message,
            face=face_out,
            limitations=LIMITATIONS,
        )

    assert gate_result.face is not None

    # --- AI detection ---------------------------------------------------
    ensemble = get_ensemble()
    ensemble.set_calibration(calibration.get("ai_ensemble"))
    try:
        ai_result = ensemble.analyze(pil_image, gate_result.face.bbox_px)
        models_out = [
            ModelScoreOut(
                name=m.name,
                ai_probability_full=m.p_ai_full,
                ai_probability_face=m.p_ai_face,
                ai_probability_combined=m.p_ai_combined,
                weight=m.weight,
                eval_auc=m.eval_auc,
            )
            for m in ai_result.models
        ]
        verdict, ai_probability, confidence_band, calibrated = (
            ai_result.verdict,
            ai_result.ai_probability,
            ai_result.confidence_band,
            ai_result.calibrated,
        )
    except Exception:
        logger.exception("AI ensemble failed")
        models_out = []
        verdict, ai_probability, confidence_band, calibrated = "uncertain", None, "low", False

    # --- reconstruction-error secondary check (only when the ensemble is
    # uncertain -- see app/detectors/reconstruction.py's module docstring
    # for why this is deliberately not just another ensemble member) ------
    reconstruction_out = None
    if verdict == "uncertain":
        try:
            recon = get_reconstruction_detector()
            recon_result = recon.analyze(pil_image, calibration.get("reconstruction_aeroblade"))
            reconstruction_out = ReconstructionCheckOut(
                reconstruction_error=recon_result.reconstruction_error,
                p_ai=recon_result.p_ai,
                verdict=recon_result.verdict,
                calibrated=recon_result.calibrated,
                note=recon_result.note,
            )
        except Exception:
            logger.exception("Reconstruction-error check failed")

    # --- heatmap (best-effort; failures shouldn't break the report) -----
    heatmap_png = None
    try:
        adapters = build_ensemble()
        if adapters:
            primary = adapters[0]
            if not hasattr(primary, "model"):
                primary.load()
            face_crop = pil_image.crop(
                (
                    gate_result.face.bbox_px[0],
                    gate_result.face.bbox_px[1],
                    gate_result.face.bbox_px[0] + gate_result.face.bbox_px[2],
                    gate_result.face.bbox_px[1] + gate_result.face.bbox_px[3],
                )
            )
            saliency = compute_saliency_map(primary, face_crop)
            heatmap_png = render_overlay_png_base64(face_crop, saliency)
    except Exception:
        logger.exception("Heatmap generation failed")

    # --- beauty engine ----------------------------------------------------
    beauty_out = None
    try:
        rgb = np.array(pil_image.convert("RGB"))
        regions = build_face_regions(gate_result.face.landmarks_norm, pil_image.width, pil_image.height)
        raw_features = compute_all_features(rgb, gate_result.face.landmarks_norm, regions)
        beauty_cal = calibration.get("beauty") or {}
        thresholds = {k: tuple(v) for k, v in beauty_cal.get("thresholds", {}).items()}
        beauty_report = score_beauty(raw_features, thresholds=thresholds or None)
        beauty_out = BeautyOut(
            score=beauty_report.score,
            level=beauty_report.level,
            subscores=beauty_report.subscores,
            raw=beauty_report.raw,
            guard_multiplier=beauty_report.guard_multiplier,
            notes=beauty_report.notes,
            calibrated=bool(beauty_cal.get("thresholds")),
        )
    except Exception:
        logger.exception("Beauty engine failed")

    # --- provenance -------------------------------------------------------
    provenance_out = None
    try:
        exif = read_exif(image_bytes)
        c2pa = read_c2pa(image_bytes)
        provenance_out = ProvenanceOut(
            exif_present=exif.exif_present,
            camera_make=exif.camera_make,
            camera_model=exif.camera_model,
            software=exif.software,
            flagged_editor_software=exif.flagged_editor,
            c2pa_present=c2pa.present,
            c2pa_claim_generator=c2pa.claim_generator,
            c2pa_is_generative_ai=c2pa.is_generative_ai,
            c2pa_actions=c2pa.actions,
        )
    except Exception:
        logger.exception("Provenance extraction failed")

    return AnalyzeReport(
        status="ok",
        message="ok",
        verdict=verdict,
        ai_probability=ai_probability,
        confidence_band=confidence_band,
        calibrated=calibrated,
        models=models_out,
        model_accuracy=_model_accuracy_out(calibration),
        beauty=beauty_out,
        face=face_out,
        provenance=provenance_out,
        heatmap_png=heatmap_png,
        reconstruction_check=reconstruction_out,
        limitations=LIMITATIONS,
    )
