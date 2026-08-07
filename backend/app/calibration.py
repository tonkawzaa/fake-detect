"""
Loads/saves app/calibration.json -- the single artifact produced by Phase 5
(scripts/evaluate.py for the AI ensemble) and consumed at API startup.

Shape:
{
  "ai_ensemble": {
    "weights": {model_name: float},
    "platt": {"a": float, "b": float} | null,
    "threshold": float,
    "per_model_auc": {model_name: float},
    "accuracy": float, "auc": float, "n": int, "out_of_fold": true,
    "per_generator": {generator_name: accuracy}
  },
  "reconstruction_aeroblade": {
    "platt": {"a": float, "b": float}, "threshold": float,
    "polarity_ok": true, "accuracy": float, "auc": float, "n": int,
    "out_of_fold": true, "per_generator": {generator_name: accuracy}
  }
}

Absence of this file is a normal, expected state before Phase 5 has been
run -- every reader here must degrade gracefully to defaults rather than
error, and the API must say "uncalibrated" rather than imply a number it
doesn't have.
"""

from __future__ import annotations

import json
from pathlib import Path

CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"


def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        return {}
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def save_calibration(data: dict) -> None:
    CALIBRATION_PATH.write_text(json.dumps(data, indent=2))
