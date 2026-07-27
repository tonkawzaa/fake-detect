"""
Phase 4: EXIF metadata.

Presence of camera EXIF is weak-to-moderate evidence of a real capture
(though social platforms strip EXIF on upload, so absence proves little).
A `Software` tag naming an editor/beauty app is a useful cross-check against
the beauty-mode engine's own (purely pixel-based) signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import piexif
from PIL import Image

KNOWN_EDITOR_SOFTWARE_HINTS = (
    "photoshop", "lightroom", "gimp", "snapseed", "facetune", "meitu",
    "beautycam", "picsart", "vsco", "ai", "midjourney", "dall-e", "stable diffusion",
    "generative", "firefly",
)


@dataclass
class ExifReport:
    exif_present: bool
    camera_make: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    software: str | None = None
    datetime_original: str | None = None
    flagged_editor: bool = False


def _decode(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore").strip("\x00 ")
        except Exception:
            return None
    return str(value).strip("\x00 ") or None


def read_exif(image_bytes: bytes) -> ExifReport:
    try:
        exif_dict = piexif.load(image_bytes)
    except Exception:
        return ExifReport(exif_present=False)

    zeroth = exif_dict.get("0th", {})
    exif_ifd = exif_dict.get("Exif", {})

    make = _decode(zeroth.get(piexif.ImageIFD.Make))
    model = _decode(zeroth.get(piexif.ImageIFD.Model))
    software = _decode(zeroth.get(piexif.ImageIFD.Software))
    lens = _decode(exif_ifd.get(piexif.ExifIFD.LensModel))
    dt_original = _decode(exif_ifd.get(piexif.ExifIFD.DateTimeOriginal))

    has_any = any([make, model, software, lens, dt_original])
    flagged = bool(software) and any(hint in software.lower() for hint in KNOWN_EDITOR_SOFTWARE_HINTS)

    return ExifReport(
        exif_present=has_any,
        camera_make=make,
        camera_model=model,
        lens_model=lens,
        software=software,
        datetime_original=dt_original,
        flagged_editor=flagged,
    )
