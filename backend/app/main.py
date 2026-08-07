from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .calibration import load_calibration
from .detectors.registry import ENSEMBLE_MODEL_NAMES
from .pipeline import LIMITATIONS, _model_accuracy_out, analyze_image
from .schemas import AnalyzeReport, ModelInfoOut

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Image Detector", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoOut)
def model_info() -> ModelInfoOut:
    calibration = load_calibration()
    ai_cal = calibration.get("ai_ensemble") or {}
    return ModelInfoOut(
        ensemble_models=list(ENSEMBLE_MODEL_NAMES),
        calibrated=ai_cal.get("platt") is not None,
        model_accuracy=_model_accuracy_out(calibration),
        limitations=LIMITATIONS,
    )


@app.post("/analyze", response_model=AnalyzeReport)
async def analyze(file: UploadFile = File(...)) -> AnalyzeReport:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 15MB)")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        return analyze_image(data)
    except Exception as e:
        logging.exception("analyze_image failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}") from e
