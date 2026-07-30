"""
Phase 1: face gate.

Requirement 1 ("เป็นภาพคน ด้วยตนเอง") is both a scope statement and an input
gate — this tool analyses portraits of people, not arbitrary images. We run
MediaPipe FaceLandmarker first and refuse to fabricate a verdict when the
input isn't a usable portrait.

Note: mediapipe 0.10.35 dropped the old `mp.solutions` API entirely (verified
empirically — `mp.solutions` no longer exists on this install). Only the
Tasks API (`mediapipe.tasks.vision`) is available, which requires a
downloaded `.task` model bundle rather than a bundled model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"

MIN_FACE_AREA_FRACTION = 0.015
MIN_FACE_BBOX_PX = 64
MANY_FACES_THRESHOLD = 5
BLUR_LOW_THRESHOLD = 60.0  # variance of Laplacian below this ~= visibly blurry


@dataclass
class FaceQuality:
    width: int
    height: int
    blur_score: float
    blur_label: str  # "sharp" | "soft" | "blurry"
    coverage: float


@dataclass
class FaceResult:
    bbox_px: tuple[int, int, int, int]  # x, y, w, h
    landmarks_norm: list[tuple[float, float, float]]  # 478 pts, normalized [0,1]


@dataclass
class GateResult:
    status: str  # "ok" | "no_face" | "low_quality"
    message: str
    face_count: int = 0
    face: FaceResult | None = None
    quality: FaceQuality | None = None
    notes: list[str] = field(default_factory=list)


class FaceGate:
    """Loads the FaceLandmarker once and reuses it across requests."""

    def __init__(self, model_path: Path = MODEL_PATH, max_num_faces: int = MANY_FACES_THRESHOLD + 5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe face_landmarker.task not found at {model_path}. "
                "Download it: curl -sL -o models/face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )
        self._mp = mp
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _blur_score(img_rgb: np.ndarray) -> float:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _bbox_from_landmarks(landmarks_norm: list[tuple[float, float, float]], width: int, height: int) -> tuple[int, int, int, int]:
        xs = [p[0] * width for p in landmarks_norm]
        ys = [p[1] * height for p in landmarks_norm]
        x0, x1 = max(min(xs), 0), min(max(xs), width)
        y0, y1 = max(min(ys), 0), min(max(ys), height)
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

    def run(self, image: Image.Image) -> GateResult:
        rgb = np.array(image.convert("RGB"))
        height, width = rgb.shape[:2]

        blur = self._blur_score(rgb)
        blur_label = "blurry" if blur < BLUR_LOW_THRESHOLD else ("soft" if blur < BLUR_LOW_THRESHOLD * 2.5 else "sharp")

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        face_count = len(result.face_landmarks)

        if face_count == 0:
            return GateResult(
                status="no_face",
                message="No face detected. This tool analyses portrait photos of people, not general images.",
                face_count=0,
            )

        # Pick the largest face by landmark bounding-box area.
        candidates = []
        for lm_list in result.face_landmarks:
            pts = [(p.x, p.y, p.z) for p in lm_list]
            bbox = self._bbox_from_landmarks(pts, width, height)
            candidates.append((bbox[2] * bbox[3], pts, bbox))
        candidates.sort(key=lambda c: c[0], reverse=True)
        _, landmarks_norm, bbox = candidates[0]

        coverage = (bbox[2] * bbox[3]) / (width * height)
        notes: list[str] = []
        if face_count > MANY_FACES_THRESHOLD:
            notes.append(f"{face_count} faces detected; analysing the largest one only.")
        elif face_count > 1:
            notes.append(f"{face_count} faces detected; analysing the largest one.")

        quality = FaceQuality(width=width, height=height, blur_score=blur, blur_label=blur_label, coverage=coverage)
        face = FaceResult(bbox_px=bbox, landmarks_norm=landmarks_norm)

        if coverage < MIN_FACE_AREA_FRACTION or bbox[2] < MIN_FACE_BBOX_PX or bbox[3] < MIN_FACE_BBOX_PX:
            return GateResult(
                status="low_quality",
                message="Face is too small in the frame for a reliable analysis.",
                face_count=face_count,
                face=face,
                quality=quality,
                notes=notes,
            )

        return GateResult(
            status="ok",
            message="ok",
            face_count=face_count,
            face=face,
            quality=quality,
            notes=notes,
        )


_gate_singleton: FaceGate | None = None


def get_face_gate() -> FaceGate:
    global _gate_singleton
    if _gate_singleton is None:
        _gate_singleton = FaceGate()
    return _gate_singleton
