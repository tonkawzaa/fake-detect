"""
Phase 5 data collection: builds data/eval/{real,ai}/ + data/eval/manifest.csv.

Two sources, deliberately combined for generator diversity (a single-generator
eval set would flatter the ensemble -- see plan's "honest limitation" note):

  1. TheKernel01/140k-Real-and-Fake-Faces -- real FFHQ photos vs. StyleGAN
     faces. Reliable, face-specific, proven in the Phase 0 smoke test.
  2. OwensLab/CommunityForensics-Eval -- the CVPR 2025 Community Forensics
     eval split, spanning many commercial/GAN/diffusion generators
     (MidjourneyV6_1, DFGAN, etc.) paired with COCO/LAION reals. This is a
     general-image dataset, not face-specific -- every label-matching
     candidate is kept directly (2026-08-06: this used to be filtered
     through face_gate to keep only usable portraits; this project no
     longer has face detection at all, so this source now contributes
     arbitrary images, matching what pipeline.py actually analyzes).

Label convention verified empirically on this dataset (do not trust field
names blindly, same discipline as the Phase 0 polarity check): label=0 rows
are the *real* source image, label=1 rows are the *generated* pairing --
`model_name` identifies which generator a given fake was produced by (and,
confusingly, tags which generator's fake a given *real* row is paired with --
it does not mean the real row itself came from that model).

Usage:
    uv run python scripts/fetch_eval_set.py
"""

from __future__ import annotations

import base64
import csv
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
REAL_DIR = EVAL_DIR / "real"
AI_DIR = EVAL_DIR / "ai"
MANIFEST_PATH = EVAL_DIR / "manifest.csv"

# Offsets used by the smoke test were 0-100; start well past that.
KERNEL01_OFFSET_START = 600
KERNEL01_TARGET_EACH = 100

# Was range(0, 30000, 2500) with length=40 per request -- that only examined
# 12 * 40 = 480 of the split's 51,836 rows (2460 of every 2500 silently
# skipped), the same class of offset-stepping bug documented in
# fetch_clip_train_set.py's CF_TRAIN_OFFSETS comment. Fixed the same way:
# step by 100 (the HF datasets-server /rows endpoint's max `length`,
# confirmed in fetch_clip_train_set.py) across the FULL split, so every row
# is actually examined. This directly caused data/eval/'s prior
# non-ffhq/stylegan AI count of 1 (a single stable_cascade sample) --
# nowhere near enough to measure generalization to unseen generators.
CF_EVAL_OFFSETS = list(range(0, 51800, 100))
CF_EVAL_TARGET_EACH = 150  # per label -- raised
# from 60 (itself barely reachable under the old, narrower offset range) so
# the diverse-generator slice of data/eval/ has enough samples (target: >=30
# non-ffhq/stylegan AI rows) to make a promotion decision statistically
# meaningful, per the DINOv3-LoRA-MAC restructuring plan.

HTTP_RETRIES = 3


def _get_json(url: str):
    last_err = None
    for _ in range(HTTP_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                import json

                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(2)
    print(f"  WARN: giving up on {url[:100]}...: {last_err}")
    return None


def fetch_kernel01(manifest: list[tuple[str, str, str]]) -> None:
    print("=== TheKernel01/140k-Real-and-Fake-Faces ===")
    real_count = ai_count = 0
    offset = KERNEL01_OFFSET_START
    while (real_count < KERNEL01_TARGET_EACH or ai_count < KERNEL01_TARGET_EACH) and offset < 5000:
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=TheKernel01%2F140k-Real-and-Fake-Faces"
            f"&config=default&split=test&offset={offset}&length=50"
        )
        d = _get_json(url)
        offset += 50
        if not d:
            continue
        for row in d.get("rows", []):
            r = row["row"]
            label = r["label"]
            src = r["image"]["src"]
            if label == 0 and real_count < KERNEL01_TARGET_EACH:
                fn = REAL_DIR / f"kernel01_{real_count:04d}.jpg"
                try:
                    urllib.request.urlretrieve(src, fn)
                except Exception as e:
                    print("  download failed:", e)
                    continue
                manifest.append((str(fn.relative_to(ROOT)), "real", "ffhq"))
                real_count += 1
            elif label == 1 and ai_count < KERNEL01_TARGET_EACH:
                fn = AI_DIR / f"kernel01_{ai_count:04d}.jpg"
                try:
                    urllib.request.urlretrieve(src, fn)
                except Exception as e:
                    print("  download failed:", e)
                    continue
                manifest.append((str(fn.relative_to(ROOT)), "ai", "stylegan"))
                ai_count += 1
        print(f"  offset={offset} real={real_count} ai={ai_count}")
    print(f"  DONE real={real_count} ai={ai_count}")


def fetch_community_forensics(manifest: list[tuple[str, str, str]]) -> None:
    print("=== OwensLab/CommunityForensics-Eval ===")
    real_kept = ai_kept = 0

    for offset in CF_EVAL_OFFSETS:
        if real_kept >= CF_EVAL_TARGET_EACH and ai_kept >= CF_EVAL_TARGET_EACH:
            break
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=OwensLab%2FCommunityForensics-Eval"
            f"&config=default&split=CompEval&offset={offset}&length=100"
        )
        d = _get_json(url)
        if not d:
            continue
        for row in d.get("rows", []):
            r = row["row"]
            label = r["label"]
            model_name = r.get("model_name") or "unknown"
            b64 = r.get("image_data")
            if not b64:
                continue
            try:
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue

            if label == 0 and real_kept < CF_EVAL_TARGET_EACH:
                fn = REAL_DIR / f"cfeval_{real_kept:04d}.jpg"
                img.save(fn, format="JPEG", quality=92)
                manifest.append((str(fn.relative_to(ROOT)), "real", r.get("real_source") or "unknown"))
                real_kept += 1
            elif label == 1 and ai_kept < CF_EVAL_TARGET_EACH:
                fn = AI_DIR / f"cfeval_{ai_kept:04d}.jpg"
                img.save(fn, format="JPEG", quality=92)
                manifest.append((str(fn.relative_to(ROOT)), "ai", model_name))
                ai_kept += 1
        print(f"  offset={offset} real_kept={real_kept}  ai_kept={ai_kept}")

    print(f"  DONE real_kept={real_kept}  ai_kept={ai_kept}")


def main() -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[tuple[str, str, str]] = []

    fetch_kernel01(manifest)
    fetch_community_forensics(manifest)

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "generator"])
        writer.writerows(manifest)

    n_real = sum(1 for _, label, _ in manifest if label == "real")
    n_ai = sum(1 for _, label, _ in manifest if label == "ai")
    generators = sorted(set(g for _, _, g in manifest))
    print(f"\nTOTAL: {len(manifest)} images ({n_real} real, {n_ai} ai)")
    print(f"Generators/sources represented: {generators}")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
