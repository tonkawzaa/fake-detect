"""
Builds a calibration set for AEROBLADEDetector (app/detectors/reconstruction.py):
real photos vs. images from LATENT-diffusion generators whose autoencoder is
the exact one app/detectors/reconstruction.py uses (SDXL's VAE, e.g. via
madebyollin/sdxl-vae-fp16-fix).

Source: OwensLab/CommunityForensics-Eval, same dataset fetch_eval_set.py and
fetch_clip_train_set.py already use, filtered to `model_name` values that
actually share that VAE. This is NOT the same filter as "diffusion model" in
general, and it is narrower than "any Stable-Diffusion-family model" too --
verified empirically by scanning the dataset's model_name values, the two
SDXL-VAE-compatible tags present are `LCM_lora_sdxl` and `LCM_lora_ssd1b`
(SSD-1B is a distilled SDXL that keeps the base model's VAE unmodified,
only the UNet was pruned). `LCM_lora_sdv15` (SD **1.5**) is deliberately
excluded even though it sounds related: SD1.5 uses a different VAE
checkpoint than SDXL (they're shape-compatible but not the same trained
weights), so an SD1.5 image has no special relationship to the SDXL VAE
this module tests against -- an earlier version of this pipeline used a
plain SD1.x VAE (stabilityai/sd-vae-ft-mse) against this same
SDXL-generated data and got the polarity assertion backwards, which is
almost certainly this exact mismatch, not AEROBLADE's method failing.
Generators that sound related but use an entirely different autoencoder
architecture are also excluded -- `stable_cascade` (Wuerstchen-style
EfficientNet latent space) and `kandinsky_2_2` (its own VAE) and
`FLUX-dev` (16-channel VAE, incompatible with SD's 4-channel one).

Because so few rows in this dataset match (~80 total across the full
51,836-row split), this calibration set is small -- expect low double or
low triple digits, not the couple hundred fetch_eval_set.py/
fetch_clip_train_set.py manage. Treat the calibration numbers this produces
as indicative, not as a tight estimate; scripts/calibrate_reconstruction.py's
polarity assertion is what actually matters here, more than the precise
accuracy number.

Never face-gated -- this dataset's SD-family generations are mostly general
scenes, not portraits (a first pass that required a detected face kept only
2 of the 80 SD-VAE-compatible AI images, nowhere near enough to calibrate),
and AEROBLADE's reconstruction-error mechanism is not face-specific at all.
2026-08-06: face detection was removed from the project entirely, so this
is no longer a special case relative to the other fetch_*.py scripts --
none of them filter on face content anymore either.

Disjoint (content-hash checked) from data/eval/ and data/clip_train/, same
guard pattern as fetch_clip_train_set.py, for data hygiene consistency
across the project's calibration sets even though AEROBLADE has no
trainable representation to leak into (its VAE and LPIPS network are both
frozen pretrained models; only a 1D Platt scale/threshold is fit).

Usage:
    uv run python scripts/fetch_aeroblade_calib_set.py
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
CLIP_TRAIN_DIR = ROOT / "data" / "clip_train"
CALIB_DIR = ROOT / "data" / "aeroblade_calib"
REAL_DIR = CALIB_DIR / "real"
AI_DIR = CALIB_DIR / "ai"
MANIFEST_PATH = CALIB_DIR / "manifest.csv"

# app/detectors/reconstruction.py's VAE is the SDXL VAE (SSD-1B is a
# distilled SDXL that keeps the base model's unmodified VAE), NOT the SD1.x
# VAE -- so "sdxl"/"ssd1b" match but a plain "SD1.5" tag (seen elsewhere in
# this dataset as e.g. "LCM_lora_sdv15") must NOT: it's a different VAE
# checkpoint, and including it would silently reintroduce the exact
# VAE-mismatch bug this filter exists to prevent. Verified empirically
# against the dataset -- see module docstring. Matching is case-insensitive
# substring; EXCLUDE_KEYWORDS wins over KEYWORDS so a future dataset
# addition like "stable_cascade_2" can't slip through the "sd"-ish
# substring net.
SD_VAE_COMPATIBLE_KEYWORDS = ("sdxl", "ssd1b", "ssd-1b", "ssd_1b")
EXCLUDE_KEYWORDS = ("cascade", "kandinsky", "flux", "sdv15", "sd15", "sd_1_5", "sd-1-5")

# Full 51,836-row split, scanned exhaustively (unlike fetch_eval_set.py's
# sparse offsets) since matching generators are rare.
CF_OFFSETS = list(range(0, 51800, 1300))
AI_TARGET = 80  # ~all SD-VAE-compatible AI images that exist in this dataset
REAL_TARGET = 80  # matched to AI_TARGET for balanced classes
HTTP_RETRIES = 3


def is_sd_vae_compatible(model_name: str) -> bool:
    name = model_name.lower()
    if any(kw in name for kw in EXCLUDE_KEYWORDS):
        return False
    return any(kw in name for kw in SD_VAE_COMPATIBLE_KEYWORDS)


def existing_hashes(*dirs: Path) -> set[str]:
    hashes = set()
    for d in dirs:
        if not d.exists():
            continue
        for p in list(d.glob("*.jpg")) + list(d.glob("*.png")):
            hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
    return hashes


def _get_json(url: str):
    last_err = None
    for _ in range(HTTP_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                import json

                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(1.5)
    return None


def main() -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[tuple[str, str, str]] = []

    print("Hashing existing data/eval/ + data/clip_train/ images to guard against overlap...")
    seen_hashes = existing_hashes(EVAL_DIR / "real", EVAL_DIR / "ai", CLIP_TRAIN_DIR / "real", CLIP_TRAIN_DIR / "ai")
    print(f"  {len(seen_hashes)} existing image hashes loaded\n")

    real_kept = ai_kept = 0
    skipped_dupes = 0
    matched_generators: set[str] = set()

    print("=== OwensLab/CommunityForensics-Eval, full scan for SD-VAE-compatible generators ===")
    for i, offset in enumerate(CF_OFFSETS):
        if ai_kept >= AI_TARGET and real_kept >= REAL_TARGET:
            break
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=OwensLab%2FCommunityForensics-Eval"
            f"&config=default&split=CompEval&offset={offset}&length=40"
        )
        d = _get_json(url)
        if not d:
            continue
        for row in d.get("rows", []):
            r = row["row"]
            label = r["label"]
            model_name = r.get("model_name") or ""
            b64 = r.get("image_data")
            if not b64:
                continue

            # AI rows: only keep SD-VAE-compatible generators.
            if label == 1:
                if not is_sd_vae_compatible(model_name) or ai_kept >= AI_TARGET:
                    continue
            elif label == 0:
                if real_kept >= REAL_TARGET:
                    continue
            else:
                continue

            try:
                raw = base64.b64decode(b64)
            except Exception:
                continue
            h = hashlib.sha256(raw).hexdigest()
            if h in seen_hashes:
                skipped_dupes += 1
                continue
            try:
                fmt = Image.open(io.BytesIO(raw)).format
            except Exception:
                continue
            seen_hashes.add(h)
            # Written verbatim, no re-encode -- AEROBLADE's signal lives in the
            # exact pixels/high-frequency residual, which a lossy JPEG re-save
            # would perturb differently depending on each row's original
            # format (this dataset mixes PNG and JPEG sources on both labels).
            ext = "png" if fmt == "PNG" else "jpg"

            if label == 1:
                matched_generators.add(model_name)
                fn = AI_DIR / f"cfaeroblade_{ai_kept:04d}.{ext}"
                fn.write_bytes(raw)
                manifest.append((str(fn.relative_to(ROOT)), "ai", model_name))
                ai_kept += 1
            else:
                fn = REAL_DIR / f"cfaeroblade_{real_kept:04d}.{ext}"
                fn.write_bytes(raw)
                manifest.append((str(fn.relative_to(ROOT)), "real", r.get("real_source") or "unknown"))
                real_kept += 1

        if (i + 1) % 5 == 0 or offset == CF_OFFSETS[-1]:
            print(
                f"  offset={offset} ai_kept={ai_kept}/{AI_TARGET}  "
                f"real_kept={real_kept}/{REAL_TARGET}  skipped_dupes={skipped_dupes}"
            )

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "generator"])
        writer.writerows(manifest)

    print(f"\nSD-VAE-compatible generators matched: {sorted(matched_generators)}")
    print(f"TOTAL: {len(manifest)} images ({real_kept} real, {ai_kept} ai)")
    if ai_kept < 15:
        print(
            "\nWARNING: very few AI images -- calibrate_reconstruction.py's polarity "
            "assertion will be low-confidence at this n. Consider widening SD_VAE_COMPATIBLE_KEYWORDS "
            "only if you've verified the new generator actually shares an SD-style VAE."
        )
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
