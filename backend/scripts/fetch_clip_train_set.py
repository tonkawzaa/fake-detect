"""
Builds a training set for CLIPLinearProbeAdapter's linear probe
(scripts/train_clip_linear_probe.py), disjoint from data/eval/ (built by
fetch_eval_set.py).

Why this needs to be a separate set, not just reusing data/eval/manifest.csv:
data/eval/manifest.csv is the held-out set scripts/evaluate.py scores the
whole ensemble against to report the "measured accuracy" shown in the UI.
train_clip_linear_probe.py's shipped weights are a direct fit (logistic
regression) on every image it's given. If that were data/eval itself, the
CLIP model's output on those images would already be partly memorized by
the time evaluate.py scores it -- not held out at all -- which would make
evaluate.py's out-of-fold accuracy dishonest for this one model specifically
(its own k-fold split doesn't help: the leakage happens one level down,
inside the probe's training, not in evaluate.py's fold logic). That is
exactly the "in-sample dressed up as out-of-sample" failure CLAUDE.md
already warns evaluate.py itself against -- this script exists so adding a
*trained* model to the ensemble doesn't reintroduce it by another path.

Same two HF sources as fetch_eval_set.py, queried at different offsets, PLUS
an exact content-hash check against every image already saved under
data/eval/{real,ai}/. The hash check is the actual, offset-independent
guarantee against overlap -- HF's datasets-server row ordering/offsets are
not a documented non-overlap contract, so don't rely on offsets alone.

Usage:
    uv run python scripts/fetch_clip_train_set.py
"""

from __future__ import annotations

import base64
import csv
import hashlib
import http.client
import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from app.detectors.face_gate import get_face_gate

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
TRAIN_DIR = ROOT / "data" / "clip_train"
REAL_DIR = TRAIN_DIR / "real"
AI_DIR = TRAIN_DIR / "ai"
MANIFEST_PATH = TRAIN_DIR / "manifest.csv"

# Chosen not to overlap fetch_eval_set.py's ranges (kernel01: 600-ish start,
# capped <5000; CF-eval: 0-30000 step 2500). The hash check below is what
# actually guarantees no overlap; these are just a head start to avoid
# wasting downloads on images we'd throw away anyway.
#
# KERNEL01_TARGET_EACH carries the bulk of the ~1000/1000 target: it's a
# dedicated, pre-cropped face dataset (FFHQ real vs. StyleGAN fake), so it's
# both the cheapest source (no face_gate needed -- see fetch_kernel01, it
# trusts the dataset's own framing) and the most reliably "a portrait" of
# the two sources. Test split has 20,000 rows total (confirmed via HF
# datasets-server /size), well-interleaved by label even at high offsets
# (spot-checked at offset=15000) -- 20000 is set as the hard cap below
# specifically because that's the split's actual size, not an arbitrary
# round number.
KERNEL01_OFFSET_START = 3000
KERNEL01_TARGET_EACH = 1000

# CF-eval is a general-image dataset (COCO/LAION reals paired with many
# commercial/GAN/diffusion generators) filtered through face_gate, so it's
# the source for generator diversity beyond StyleGAN -- at the cost of a
# low portrait-yield rate (6% real / 2% ai observed on the first 150-per-label
# scan: 9/150 real, 3/150 ai kept after face-gating).
#
# CF_TRAIN_OFFSETS steps by 100 = the HF datasets-server /rows endpoint's max
# `length` (confirmed empirically: length=200 is rejected with "must not be
# greater than 100") -- offset step must equal length for a contiguous scan.
# An earlier version of this constant stepped by 400 while only fetching
# length=40 per request, silently skipping 360 of every 400 rows -- that,
# combined with the target-vs-seen bug below, is why the first run plateaued
# at 9/3 kept instead of anywhere near CF_TRAIN_TARGET_EACH.
CF_TRAIN_OFFSETS = list(range(0, 51800, 100))

# This is a KEPT-count target (see fetch_community_forensics: the stopping
# condition below now checks real_kept/ai_kept, not how many rows were
# merely examined) -- reaching it requires scanning roughly kept/yield_rate
# candidates per label. At the ~2% ai yield rate observed, 100 kept ai needs
# ~5,000 ai-labeled rows examined -- well within the 51,836-row split, so
# this (unlike an earlier 310-each attempt) should reliably hit its target
# rather than being capped by the split running out. Lowered from 310 to
# 100 per explicit request, trading some generator-diversity breadth for a
# much shorter scan.
CF_TRAIN_TARGET_EACH = 100

HTTP_RETRIES = 3


def existing_hashes(*dirs: Path) -> set[str]:
    hashes = set()
    for d in dirs:
        if not d.exists():
            continue
        for p in list(d.glob("*.jpg")) + list(d.glob("*.png")):
            hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
    return hashes


def _get_json(url: str):
    """Some CF-eval rows/pages are large (base64 image data inline, up to
    100 rows/request) -- a mid-transfer network hiccup on one of these can
    raise http.client.IncompleteRead rather than a clean HTTPError/URLError,
    which used to escape this function's retry loop entirely and crash the
    whole script (observed: crashed at offset 2800 having read 170MB of a
    ~200MB response, losing all not-yet-written progress on a scan that
    was, by design, going to take a long time to reach its target)."""
    last_err = None
    for _ in range(HTTP_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                import json

                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, http.client.HTTPException) as e:
            last_err = e
            time.sleep(2)
    print(f"  WARN: giving up on {url[:100]}...: {last_err}")
    return None


def fetch_kernel01(manifest: list[tuple[str, str, str]], seen_hashes: set[str]) -> None:
    print("=== TheKernel01/140k-Real-and-Fake-Faces (train split) ===")
    real_count = ai_count = 0
    skipped_dupes = 0
    offset = KERNEL01_OFFSET_START
    while (real_count < KERNEL01_TARGET_EACH or ai_count < KERNEL01_TARGET_EACH) and offset < 20000:
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
            want_real = label == 0 and real_count < KERNEL01_TARGET_EACH
            want_ai = label == 1 and ai_count < KERNEL01_TARGET_EACH
            if not (want_real or want_ai):
                continue
            try:
                with urllib.request.urlopen(src, timeout=30) as resp:
                    raw = resp.read()
            except Exception as e:
                print("  download failed:", e)
                continue
            h = hashlib.sha256(raw).hexdigest()
            if h in seen_hashes:
                skipped_dupes += 1
                continue
            seen_hashes.add(h)
            if want_real:
                fn = REAL_DIR / f"kernel01_train_{real_count:04d}.jpg"
                fn.write_bytes(raw)
                manifest.append((str(fn.relative_to(ROOT)), "real", "ffhq"))
                real_count += 1
            else:
                fn = AI_DIR / f"kernel01_train_{ai_count:04d}.jpg"
                fn.write_bytes(raw)
                manifest.append((str(fn.relative_to(ROOT)), "ai", "stylegan"))
                ai_count += 1
        print(f"  offset={offset} real={real_count} ai={ai_count} skipped_dupes={skipped_dupes}")
    print(f"  DONE real={real_count} ai={ai_count} (skipped {skipped_dupes} images already in data/eval or seen twice)")


def fetch_community_forensics(manifest: list[tuple[str, str, str]], seen_hashes: set[str]) -> None:
    print("=== OwensLab/CommunityForensics-Eval (train split, face-gated) ===")
    gate = get_face_gate()
    real_count = ai_count = 0
    real_kept = ai_kept = 0
    skipped_dupes = 0

    for offset in CF_TRAIN_OFFSETS:
        if real_kept >= CF_TRAIN_TARGET_EACH and ai_kept >= CF_TRAIN_TARGET_EACH:
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
            except Exception:
                continue
            h = hashlib.sha256(raw).hexdigest()
            if h in seen_hashes:
                skipped_dupes += 1
                continue

            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue

            if label == 0 and real_kept < CF_TRAIN_TARGET_EACH:
                gate_result = gate.run(img)
                real_count += 1
                if gate_result.status == "ok":
                    seen_hashes.add(h)
                    fn = REAL_DIR / f"cftrain_{real_kept:04d}.jpg"
                    img.save(fn, format="JPEG", quality=92)
                    manifest.append((str(fn.relative_to(ROOT)), "real", r.get("real_source") or "unknown"))
                    real_kept += 1
            elif label == 1 and ai_kept < CF_TRAIN_TARGET_EACH:
                gate_result = gate.run(img)
                ai_count += 1
                if gate_result.status == "ok":
                    seen_hashes.add(h)
                    fn = AI_DIR / f"cftrain_{ai_kept:04d}.jpg"
                    img.save(fn, format="JPEG", quality=92)
                    manifest.append((str(fn.relative_to(ROOT)), "ai", model_name))
                    ai_kept += 1
        print(
            f"  offset={offset} real_seen={real_count} real_kept={real_kept}  "
            f"ai_seen={ai_count} ai_kept={ai_kept}  skipped_dupes={skipped_dupes}"
        )

    print(f"  DONE real_kept={real_kept} (of {real_count} seen)  ai_kept={ai_kept} (of {ai_count} seen)")


def main() -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[tuple[str, str, str]] = []

    print("Hashing existing data/eval/ images to guard against overlap...")
    seen_hashes = existing_hashes(EVAL_DIR / "real", EVAL_DIR / "ai")
    print(f"  {len(seen_hashes)} existing eval-set image hashes loaded\n")

    fetch_kernel01(manifest, seen_hashes)
    fetch_community_forensics(manifest, seen_hashes)

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
