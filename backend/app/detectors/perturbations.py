"""
Approximates vincentlc's "perturbation-aware training" augmentation
pipeline (NTIRE 2026, "Robust AI-Generated Image Detection via SigLIP2-Giant
and Perturbation-Aware Training"). Originally written for a since-deleted
scripts/train_siglip2_probe.py; recovered from git history (commit
e14f0b6's parent) to serve the same role in
scripts/train_dinov3_lora_mac_stage2.py's self-distillation stage -- the
perturbation logic below is unchanged from the original.

NTIRE 2026's own official distortion pipeline -- the one actually used to
train vincentlc's submission and to score the challenge's "robust ROC AUC"
metric -- is not published/available to this repo. This module is an
independently-written stand-in inspired only by their method description
("every training image is distorted, with up to three random
post-processing operations sampled at multiple severity levels [5
levels]"), using a reasonable but different set of concrete operations
(JPEG recompression, Gaussian blur, Gaussian noise, downscale/upscale).
Treat any robustness improvement from training with this module as
evidence for the general approach (perturb inputs -> re-extract features ->
train against them), not as reproducing vincentlc's actual reported
robust-AUC numbers -- those were measured against the specific official
pipeline, which this is not.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter

NUM_SEVERITY_LEVELS = 5
MAX_OPS_PER_IMAGE = 3


def _jpeg_recompress(image: Image.Image, severity: int) -> Image.Image:
    quality = max(90 - severity * 15, 5)  # severity 1..5 -> quality 75..5
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _gaussian_blur(image: Image.Image, severity: int) -> Image.Image:
    radius = severity * 0.8  # severity 1..5 -> radius 0.8..4.0
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _gaussian_noise(image: Image.Image, severity: int) -> Image.Image:
    std = severity * 6.0  # severity 1..5 -> std 6..30 (pixel scale 0-255)
    arr = np.array(image.convert("RGB")).astype(np.float32)
    noisy = np.clip(arr + np.random.normal(0, std, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def _resize_degrade(image: Image.Image, severity: int) -> Image.Image:
    scale = 1.0 - severity * 0.12  # severity 1..5 -> downscale to 88%..40%, then back up
    w, h = image.size
    small = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


_OPS = (_jpeg_recompress, _gaussian_blur, _gaussian_noise, _resize_degrade)


def random_perturb(image: Image.Image, rng: random.Random) -> Image.Image:
    """Applies 1..MAX_OPS_PER_IMAGE random distortions at random severity
    levels, in random order -- same shape as vincentlc's description ("up
    to three random post-processing operations ... multiple severity
    levels"), using this module's own operation set. `rng` is caller-owned
    so training runs are reproducible given a fixed seed."""
    ops = rng.sample(_OPS, rng.randint(1, MAX_OPS_PER_IMAGE))
    out = image
    for op in ops:
        out = op(out, rng.randint(1, NUM_SEVERITY_LEVELS))
    return out
