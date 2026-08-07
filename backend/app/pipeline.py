"""
Orchestrates one /analyze request: AI ensemble + provenance -> fused report.
Each analysis stage is independent (a failure in provenance parsing
shouldn't take down the AI verdict), so stages are wrapped defensively and
degrade to "unavailable" rather than 500ing the whole request.

2026-08-06: face detection (face_gate, MediaPipe FaceLandmarker) and the
beauty/retouching-filter engine were removed from the project entirely, per
explicit request. This tool analyzes any image, full-frame only -- there is
no face-crop pass for the AI-detection ensemble anymore and no beauty score.
AnalyzeReport.status is always "ok" for any image this function finishes
analyzing.
"""

from __future__ import annotations

import io
import logging

from PIL import Image

from .calibration import load_calibration
from .detectors.ai_detector import get_ensemble
from .detectors.heatmap import compute_saliency_map, render_overlay_png_base64
from .detectors.reconstruction import get_reconstruction_detector
from .detectors.registry import build_ensemble
from .forensics.credentials import read_c2pa
from .forensics.metadata import read_exif, read_xmp
from .schemas import (
    AnalyzeReport,
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

    # --- AI detection -----------------------------------------------------
    ensemble = get_ensemble()
    ensemble.set_calibration(calibration.get("ai_ensemble"))
    try:
        ai_result = ensemble.analyze(pil_image)
        models_out = [
            ModelScoreOut(
                name=m.name,
                ai_probability=m.p_ai,
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

    # --- provenance ---------------------------------------------------
    # Computed here (ahead of the reconstruction-error gate below) because a
    # C2PA manifest -- or a bare XMP `Iptc4xmpExt:DigitalSourceType` tag,
    # see app/forensics/metadata.py's module docstring -- naming a
    # generative-AI producer is treated as near-conclusive and overrides the
    # ensemble's verdict outright, so an "uncertain" ensemble verdict on a
    # provably-AI image shouldn't still trigger the expensive AEROBLADE
    # secondary check below. C2PA is checked first: it's cryptographically
    # signed, whereas a bare XMP tag is just rewritable metadata with no
    # such guarantee.
    provenance_out = None
    verdict_source = "ensemble"
    try:
        exif = read_exif(image_bytes)
        c2pa = read_c2pa(image_bytes)
        xmp = read_xmp(image_bytes)
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
            xmp_present=xmp.present,
            xmp_digital_source_type=xmp.digital_source_type,
            xmp_is_generative_ai=xmp.is_generative_ai,
        )
        if c2pa.is_generative_ai:
            verdict, ai_probability, confidence_band = "likely_ai", 1.0, "high"
            verdict_source = "c2pa"
        elif xmp.is_generative_ai:
            verdict, ai_probability, confidence_band = "likely_ai", 1.0, "high"
            verdict_source = "xmp"
    except Exception:
        logger.exception("Provenance extraction failed")

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
            saliency = compute_saliency_map(primary, pil_image)
            heatmap_png = render_overlay_png_base64(pil_image, saliency)
    except Exception:
        logger.exception("Heatmap generation failed")

    return AnalyzeReport(
        status="ok",
        message="ok",
        verdict=verdict,
        ai_probability=ai_probability,
        confidence_band=confidence_band,
        verdict_source=verdict_source,
        calibrated=calibrated,
        models=models_out,
        model_accuracy=_model_accuracy_out(calibration),
        provenance=provenance_out,
        heatmap_png=heatmap_png,
        reconstruction_check=reconstruction_out,
        limitations=LIMITATIONS,
    )
