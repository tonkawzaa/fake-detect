"""
Phase 3: face regions from FaceMesh landmarks.

Rather than hand-picking landmark-index lists for "forehead"/"cheeks"/"chin"
(easy to get subtly wrong), we build regions from MediaPipe's own named
connection sets (FACE_LANDMARKS_FACE_OVAL / LEFT_EYE / RIGHT_EYE / EYEBROWS /
LIPS), each of which is a cycle of edges we walk into an ordered polygon.

skin region  = face oval, minus (eyes + eyebrows + lips), dilated a bit so
               the exclusion has clean margins.
eye region   = the reference "known-sharp" region: eyelashes stay sharp under
               a skin-smoothing filter, so a skin/eye sharpness ratio is
               informative regardless of hairstyle.
border region = an image-edge strip, outside any face landmark, used as a
               second sharp/noise reference that's always available (hair
               region is not, e.g. for bald subjects or tight crops) --
               a deliberate simplification vs. a full hair segmentation model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BORDER_STRIP_FRACTION = 0.06


def _ordered_loop(connections, index_map: dict[int, int] | None = None) -> list[int]:
    """Walk a MediaPipe Connection list (unordered edges forming one cycle)
    into an ordered list of landmark indices."""
    adjacency: dict[int, list[int]] = {}
    for c in connections:
        adjacency.setdefault(c.start, []).append(c.end)
        adjacency.setdefault(c.end, []).append(c.start)

    start = next(iter(adjacency))
    loop = [start]
    prev = None
    cur = start
    while True:
        neighbors = [n for n in adjacency[cur] if n != prev]
        nxt = None
        for n in neighbors:
            if n not in loop:
                nxt = n
                break
        if nxt is None:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    return loop


def _polygon_px(landmarks_norm: list[tuple[float, float, float]], indices: list[int], width: int, height: int) -> np.ndarray:
    pts = [(landmarks_norm[i][0] * width, landmarks_norm[i][1] * height) for i in indices]
    return np.array(pts, dtype=np.int32)


@dataclass
class FaceRegions:
    skin_mask: np.ndarray  # bool, HxW
    eye_mask: np.ndarray
    lips_mask: np.ndarray
    border_mask: np.ndarray
    face_bbox: tuple[int, int, int, int]


def build_face_regions(landmarks_norm: list[tuple[float, float, float]], width: int, height: int) -> FaceRegions:
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

    oval_idx = _ordered_loop(C.FACE_LANDMARKS_FACE_OVAL)
    left_eye_idx = _ordered_loop(C.FACE_LANDMARKS_LEFT_EYE)
    right_eye_idx = _ordered_loop(C.FACE_LANDMARKS_RIGHT_EYE)
    left_brow_idx = _ordered_loop(C.FACE_LANDMARKS_LEFT_EYEBROW)
    right_brow_idx = _ordered_loop(C.FACE_LANDMARKS_RIGHT_EYEBROW)
    lips_idx = _ordered_loop(C.FACE_LANDMARKS_LIPS)

    face_poly = _polygon_px(landmarks_norm, oval_idx, width, height)
    left_eye_poly = _polygon_px(landmarks_norm, left_eye_idx, width, height)
    right_eye_poly = _polygon_px(landmarks_norm, right_eye_idx, width, height)
    left_brow_poly = _polygon_px(landmarks_norm, left_brow_idx, width, height)
    right_brow_poly = _polygon_px(landmarks_norm, right_brow_idx, width, height)
    lips_poly = _polygon_px(landmarks_norm, lips_idx, width, height)

    face_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(face_mask, [face_poly], 1)

    exclude_mask = np.zeros((height, width), dtype=np.uint8)
    for poly in (left_eye_poly, right_eye_poly, left_brow_poly, right_brow_poly, lips_poly):
        cv2.fillPoly(exclude_mask, [poly], 1)
    kernel = np.ones((9, 9), np.uint8)
    exclude_mask = cv2.dilate(exclude_mask, kernel, iterations=1)

    skin_mask = (face_mask.astype(bool)) & (~exclude_mask.astype(bool))

    eye_mask_u8 = np.zeros((height, width), dtype=np.uint8)
    for poly in (left_eye_poly, right_eye_poly):
        cv2.fillPoly(eye_mask_u8, [poly], 1)
    eye_mask_u8 = cv2.dilate(eye_mask_u8, np.ones((5, 5), np.uint8), iterations=1)
    eye_mask = eye_mask_u8.astype(bool)

    lips_mask_u8 = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(lips_mask_u8, [lips_poly], 1)
    lips_mask = lips_mask_u8.astype(bool)

    border_mask = np.zeros((height, width), dtype=bool)
    bw = max(1, int(width * BORDER_STRIP_FRACTION))
    bh = max(1, int(height * BORDER_STRIP_FRACTION))
    border_mask[:bh, :] = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True
    # never treat pixels inside the face oval as "border reference"
    border_mask &= ~face_mask.astype(bool)

    x, y, w, h = cv2.boundingRect(face_poly)
    return FaceRegions(
        skin_mask=skin_mask,
        eye_mask=eye_mask,
        lips_mask=lips_mask,
        border_mask=border_mask,
        face_bbox=(x, y, w, h),
    )
