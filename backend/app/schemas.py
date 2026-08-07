from __future__ import annotations

from pydantic import BaseModel


class ModelScoreOut(BaseModel):
    name: str
    ai_probability: float
    weight: float
    eval_auc: float | None = None


class ModelAccuracyOut(BaseModel):
    """Measured on the held-out eval set (scripts/evaluate.py), NOT the
    per-image confidence above -- see the two-percentages note in the plan."""

    accuracy: float | None = None
    auc: float | None = None
    n: int | None = None
    out_of_fold: bool = False
    per_generator: dict[str, float] | None = None
    note: str | None = None


class ReconstructionCheckOut(BaseModel):
    """AEROBLADE reconstruction-error secondary check (app/detectors/reconstruction.py).
    Only ever populated when the main ensemble's verdict was "uncertain" --
    see pipeline.py. Scope is deliberately narrow: meaningful only for
    Stable-Diffusion-family latent-diffusion images, not GANs or other
    generator architectures -- see note."""

    reconstruction_error: float
    p_ai: float | None = None
    verdict: str
    calibrated: bool
    note: str


class ProvenanceOut(BaseModel):
    exif_present: bool
    camera_make: str | None = None
    camera_model: str | None = None
    software: str | None = None
    flagged_editor_software: bool = False
    c2pa_present: bool = False
    c2pa_claim_generator: str | None = None
    c2pa_is_generative_ai: bool = False
    c2pa_actions: list[str] = []
    xmp_present: bool = False
    xmp_digital_source_type: str | None = None
    xmp_is_generative_ai: bool = False


class AnalyzeReport(BaseModel):
    status: str  # always "ok" -- kept (rather than dropped) so existing API
    # consumers don't break on a missing key, but nothing sets it to anything
    # but "ok" anymore.
    message: str
    verdict: str | None = None
    ai_probability: float | None = None
    confidence_band: str | None = None
    calibrated: bool = False
    verdict_source: str = "ensemble"  # "ensemble" | "c2pa" -- see pipeline.py
    models: list[ModelScoreOut] = []
    model_accuracy: ModelAccuracyOut | None = None
    provenance: ProvenanceOut | None = None
    heatmap_png: str | None = None
    reconstruction_check: ReconstructionCheckOut | None = None
    limitations: list[str] = []


class ModelInfoOut(BaseModel):
    ensemble_models: list[str]
    calibrated: bool
    model_accuracy: ModelAccuracyOut | None = None
    limitations: list[str]
