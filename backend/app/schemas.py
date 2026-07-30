from __future__ import annotations

from pydantic import BaseModel


class ModelScoreOut(BaseModel):
    name: str
    ai_probability_full: float
    ai_probability_face: float | None = None  # None when no usable face was found (see pipeline.py)
    ai_probability_combined: float
    weight: float
    eval_auc: float | None = None


class ModelAccuracyOut(BaseModel):
    """Measured on the held-out eval set (scripts/evaluate.py), NOT the
    per-image confidence above -- see the two-percentages note in the plan.
    data/eval/ is itself face-gated by construction (scripts/fetch_eval_set.py
    keeps only images that pass face_gate), so this accuracy is a claim about
    portraits specifically -- it doesn't cover the no-face path pipeline.py
    now also analyzes (see pipeline.py's module docstring, 2026-07-30)."""

    accuracy: float | None = None
    auc: float | None = None
    n: int | None = None
    out_of_fold: bool = False
    per_generator: dict[str, float] | None = None
    note: str | None = None


class BeautyOut(BaseModel):
    score: float
    level: str
    subscores: dict[str, float]
    raw: dict[str, float]
    guard_multiplier: float
    notes: list[str]
    calibrated: bool


class FaceQualityOut(BaseModel):
    width: int
    height: int
    blur_score: float
    blur_label: str
    coverage: float


class FaceOut(BaseModel):
    count: int
    bbox: tuple[int, int, int, int] | None = None
    quality: FaceQualityOut | None = None
    notes: list[str] = []


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
    status: str  # always "ok" as of 2026-07-30 -- face_gate no longer gates
    # the request (see pipeline.py's module docstring); this field is kept
    # (rather than dropped) so existing API consumers don't break on a
    # missing key, but nothing sets it to anything but "ok" anymore. Whether
    # a face was found is now informational-only, in `face` below.
    message: str
    verdict: str | None = None
    ai_probability: float | None = None
    confidence_band: str | None = None
    calibrated: bool = False
    verdict_source: str = "ensemble"  # "ensemble" | "c2pa" -- see pipeline.py
    models: list[ModelScoreOut] = []
    model_accuracy: ModelAccuracyOut | None = None
    beauty: BeautyOut | None = None
    face: FaceOut
    provenance: ProvenanceOut | None = None
    heatmap_png: str | None = None
    reconstruction_check: ReconstructionCheckOut | None = None
    limitations: list[str] = []


class ModelInfoOut(BaseModel):
    ensemble_models: list[str]
    calibrated: bool
    model_accuracy: ModelAccuracyOut | None = None
    beauty_calibrated: bool
    limitations: list[str]
