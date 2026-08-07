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
# dedicated, pre-cropped face dataset (FFHQ real vs. StyleGAN fake). Test
# split has 20,000 rows total (confirmed via HF datasets-server /size),
# well-interleaved by label even at high offsets (spot-checked at
# offset=15000) -- 20000 is set as the hard cap below specifically because
# that's the split's actual size, not an arbitrary round number.
KERNEL01_OFFSET_START = 3000
KERNEL01_TARGET_EACH = 1000

# CF-eval is a general-image dataset (COCO/LAION reals paired with many
# commercial/GAN/diffusion generators), the source for generator diversity
# beyond StyleGAN. 2026-08-06: this used to be filtered through face_gate
# (kept only if a usable portrait was detected), which this project no
# longer has -- every label-matching candidate is now kept directly, so
# real_kept/ai_kept below are just running counts, not a "kept of seen"
# yield.
#
# CF_TRAIN_OFFSETS steps by 100 = the HF datasets-server /rows endpoint's max
# `length` (confirmed empirically: length=200 is rejected with "must not be
# greater than 100") -- offset step must equal length for a contiguous scan.
CF_TRAIN_OFFSETS = list(range(0, 51800, 100))

# Raised back up from 100 (itself a deliberate reduction from an original
# 310, "trading generator-diversity breadth for a much shorter scan") for
# the DINOv3-LoRA-MAC restructuring: a LoRA fine-tune and a generator-family
# MAC aux head both need real generator diversity to be anything other than
# "is it StyleGAN" in disguise (see CLAUDE.md's account of why the
# pre-diversification data/clip_train/ -- 99.4% ffhq/stylegan, n=2012 -- was
# a load-bearing blocker for that restructuring).
#
# 2026-08-06: with face_gate filtering removed, every label-matching row is
# kept, so this target used to be satisfied almost immediately -- in
# practice, entirely by DFGAN + StyleGAN from the first ~1500 rows of the
# split, silently reintroducing the exact narrowness the 2026-08-01
# diversification effort existed to fix (measured: a first face_gate-free
# run produced ai rows that were 100% DFGAN/StyleGAN, zero diffusion-model
# generators). CF_TRAIN_PER_GENERATOR_CAP below forces the scan to keep
# going deeper into the split rather than stopping as soon as the first
# couple of generators fill the quota -- at cap=150, reaching
# CF_TRAIN_TARGET_EACH=500 per label requires rows from at least 4 distinct
# generators, so diversity is structural rather than incidental. This
# necessarily means scanning much further into the 51,836-row split than
# the pre-cap version did (slower, more HTTP requests), which is the
# deliberate trade this makes.
CF_TRAIN_TARGET_EACH = 500
CF_TRAIN_PER_GENERATOR_CAP = 150

HTTP_RETRIES = 3
RATE_LIMIT_RETRIES = 6  # see _get_json's docstring -- 429s get their own, longer backoff budget


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
    was, by design, going to take a long time to reach its target).

    2026-08-06: HTTP 429 (rate limit) needs its own, much longer backoff --
    a flat 2s retry (fine for transient network blips) just re-triggers the
    same 429 immediately under sustained rate limiting, observed to burn
    through dozens of offsets making zero progress once HF started
    throttling this unauthenticated client (see CF_TRAIN_PER_GENERATOR_CAP's
    deeper scan above, which issues far more requests than the pre-cap
    version did). Honors a `Retry-After` header when HF sends one, else
    backs off exponentially (5s, 10s, 20s, ...) up to RATE_LIMIT_RETRIES
    attempts specifically for 429s, separate from HTTP_RETRIES for other
    transient errors."""
    last_err = None
    generic_attempt = 0
    rate_limit_attempt = 0
    while generic_attempt < HTTP_RETRIES and rate_limit_attempt < RATE_LIMIT_RETRIES:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                import json

                return json.load(resp)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if retry_after else min(5 * (2**rate_limit_attempt), 60)
                rate_limit_attempt += 1
                time.sleep(wait)
            else:
                generic_attempt += 1
                time.sleep(2)
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException) as e:
            last_err = e
            generic_attempt += 1
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
    print("=== OwensLab/CommunityForensics-Eval (train split) ===")
    real_kept = ai_kept = 0
    skipped_dupes = 0
    real_gen_counts: dict[str, int] = {}
    ai_gen_counts: dict[str, int] = {}

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
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue

            # Hash the RE-ENCODED bytes (same format/quality this function
            # saves with), NOT the raw pre-decode bytes from the HF source --
            # every image from this source gets decoded and re-saved as
            # JPEG q=92 rather than written verbatim (unlike kernel01's
            # byte-for-byte saves below), so existing_hashes()'s on-disk
            # hashes (computed from data/eval/'s already-saved, already
            # re-encoded files) would almost never match a hash of the
            # pre-encode raw bytes for the same source image -- confirmed
            # empirically: this exact mismatch let 299 images end up
            # duplicated in both data/eval/ and data/clip_train/ despite
            # skipped_dupes reporting only 2, before this fix. Hashing (and
            # saving) the identical re-encoded bytes here makes the
            # comparison apples-to-apples.
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            encoded = buf.getvalue()
            h = hashlib.sha256(encoded).hexdigest()
            if h in seen_hashes:
                skipped_dupes += 1
                continue

            if label == 0 and real_kept < CF_TRAIN_TARGET_EACH:
                gen = r.get("real_source") or "unknown"
                if real_gen_counts.get(gen, 0) >= CF_TRAIN_PER_GENERATOR_CAP:
                    continue
                seen_hashes.add(h)
                fn = REAL_DIR / f"cftrain_{real_kept:04d}.jpg"
                fn.write_bytes(encoded)
                manifest.append((str(fn.relative_to(ROOT)), "real", gen))
                real_kept += 1
                real_gen_counts[gen] = real_gen_counts.get(gen, 0) + 1
            elif label == 1 and ai_kept < CF_TRAIN_TARGET_EACH:
                if ai_gen_counts.get(model_name, 0) >= CF_TRAIN_PER_GENERATOR_CAP:
                    continue
                seen_hashes.add(h)
                fn = AI_DIR / f"cftrain_{ai_kept:04d}.jpg"
                fn.write_bytes(encoded)
                manifest.append((str(fn.relative_to(ROOT)), "ai", model_name))
                ai_kept += 1
                ai_gen_counts[model_name] = ai_gen_counts.get(model_name, 0) + 1
        print(
            f"  offset={offset} real_kept={real_kept}  ai_kept={ai_kept}  skipped_dupes={skipped_dupes}  "
            f"ai_generators={dict(sorted(ai_gen_counts.items()))}"
        )

    print(f"  DONE real_kept={real_kept}  ai_kept={ai_kept}")
    print(f"  real generator breakdown: {dict(sorted(real_gen_counts.items()))}")
    print(f"  ai generator breakdown: {dict(sorted(ai_gen_counts.items()))}")


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
