"""
Phase 4: EXIF + raw XMP metadata.

Presence of camera EXIF is weak-to-moderate evidence of a real capture
(though social platforms strip EXIF on upload, so absence proves little).
A `Software` tag naming a known editor/beauty app is a useful independent
signal that the image went through post-processing.

Separately, `read_xmp` reads the IPTC `Iptc4xmpExt:DigitalSourceType` property
straight out of an image's embedded XMP packet -- the same IPTC NewsCode
(http://cv.iptc.org/newscodes/digitalsourcetype/*) that C2PA action
assertions carry (see credentials.py), but here with no C2PA manifest
involved at all. Some export paths write this IPTC field to XMP without ever
attaching a signed C2PA manifest, so this is a fallback signal C2PA parsing
alone would miss. Unlike a signed C2PA manifest, a bare XMP tag has no
cryptographic guarantee behind it -- it's just metadata, trivially rewritable
by any tool (exiftool included) -- so it's a weaker claim than a valid C2PA
signature, even though this app currently treats both as equally conclusive
for the "force 100% AI" override in pipeline.py.

Parsed with a regex over the raw XMP packet rather than a full RDF/XML
parser -- this repo has no XMP dependency yet and the property has few
serialization shapes in practice (attribute, element text, or rdf:resource).
A malformed or unusual serialization is treated the same as "not found",
consistent with this module's existing best-effort EXIF handling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import piexif
from PIL import Image

from .ai_hints import GENERATIVE_AI_HINTS

KNOWN_EDITOR_SOFTWARE_HINTS = (
    "photoshop", "lightroom", "gimp", "snapseed", "facetune", "meitu",
    "beautycam", "picsart", "vsco", "ai", "midjourney", "dall-e", "stable diffusion",
    "generative", "firefly",
)

_XMP_PACKET_RE = re.compile(rb"<\?xpacket begin=.*?<\?xpacket end=[^>]*\?>", re.DOTALL)
_DIGITAL_SOURCE_TYPE_RE = re.compile(
    rb"Iptc4xmpExt:DigitalSourceType"
    rb"(?:\s+rdf:resource=[\"']([^\"']+)[\"']"  # <Iptc4xmpExt:DigitalSourceType rdf:resource="..."/>
    rb"|>([^<]+)<"  # <Iptc4xmpExt:DigitalSourceType>...</Iptc4xmpExt:DigitalSourceType>
    rb"|=[\"']([^\"']+)[\"'])",  # Iptc4xmpExt:DigitalSourceType="..." (compact attribute form)
    re.IGNORECASE,
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


@dataclass
class XmpReport:
    present: bool
    digital_source_type: str | None = None
    is_generative_ai: bool = False


def read_xmp(image_bytes: bytes) -> XmpReport:
    packet_match = _XMP_PACKET_RE.search(image_bytes)
    if not packet_match:
        return XmpReport(present=False)

    dst_match = _DIGITAL_SOURCE_TYPE_RE.search(packet_match.group(0))
    if not dst_match:
        return XmpReport(present=True)

    raw_value = next(g for g in dst_match.groups() if g is not None)
    digital_source_type = raw_value.decode("utf-8", errors="ignore").strip()
    is_ai = any(hint in digital_source_type.lower() for hint in GENERATIVE_AI_HINTS)

    return XmpReport(
        present=True,
        digital_source_type=digital_source_type,
        is_generative_ai=is_ai,
    )
