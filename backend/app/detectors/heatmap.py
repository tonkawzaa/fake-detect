"""
Heatmap overlay showing where the AI-detector's decision came from.

Note on method: the plan sketched ViT attention rollout, but timm's fused
attention path (used by both surviving models) doesn't expose per-head
attention weights without invasive monkeypatching of internals that would be
fragile across library versions. We use vanilla input-gradient saliency
instead (gradient of the AI-logit w.r.t. input pixels) -- it is model-agnostic,
robust to library updates, and answers the same user question ("where did the
model look?") without the fragility. Labelled "saliency" in the API, not
"attention", so we're not claiming a method we didn't implement.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image

from .registry import CommForAdapter, DINOv3LoRAMACAdapter, HFClassifierAdapter, DetectorAdapter


def _saliency_dinov3_lora_mac(adapter: DINOv3LoRAMACAdapter, image: Image.Image) -> np.ndarray:
    """Gradient of the MAIN branch logit only, never a combined/summed MAC
    output -- the aux (generator-family) branch is a training-time
    regularizer signal, not something a user asking "where did the model
    look" should see blended in."""
    x = adapter.model.transform(image.convert("RGB")).unsqueeze(0).to(adapter.device)
    x.requires_grad_(True)
    main_logit, _aux_logit = adapter.model(x)
    logit = main_logit.squeeze()
    adapter.model.zero_grad(set_to_none=True)
    logit.backward()
    grad = x.grad.detach().abs().squeeze(0)  # (3, H, W)
    return grad.max(dim=0).values.cpu().numpy()


def _saliency_commfor(adapter: CommForAdapter, image: Image.Image) -> np.ndarray:
    x = adapter.transform(image.convert("RGB")).unsqueeze(0).to(adapter.device)
    x.requires_grad_(True)
    out = adapter.model(x)
    logit = out.squeeze()
    adapter.model.zero_grad(set_to_none=True)
    logit.backward()
    grad = x.grad.detach().abs().squeeze(0)  # (3, H, W)
    return grad.max(dim=0).values.cpu().numpy()


def _saliency_hf_classifier(adapter: HFClassifierAdapter, image: Image.Image) -> np.ndarray:
    inputs = adapter.processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(adapter.device)
    pixel_values.requires_grad_(True)
    inputs["pixel_values"] = pixel_values
    logits = adapter.model(**inputs).logits
    target = logits[0, adapter.fake_idx]
    adapter.model.zero_grad(set_to_none=True)
    target.backward()
    grad = pixel_values.grad.detach().abs().squeeze(0)  # (3, H, W)
    return grad.max(dim=0).values.cpu().numpy()


def compute_saliency_map(adapter: DetectorAdapter, image: Image.Image) -> np.ndarray:
    """Returns a float32 array, same H/W as the model's input tensor, values >= 0."""
    was_training = getattr(adapter, "model", None) is not None and adapter.model.training
    try:
        if isinstance(adapter, DINOv3LoRAMACAdapter):
            sal = _saliency_dinov3_lora_mac(adapter, image)
        elif isinstance(adapter, CommForAdapter):
            sal = _saliency_commfor(adapter, image)
        elif isinstance(adapter, HFClassifierAdapter):
            sal = _saliency_hf_classifier(adapter, image)
        else:
            raise TypeError(f"No saliency method for adapter type {type(adapter)}")
    finally:
        if was_training:
            adapter.model.train()
    return sal


def render_overlay_png_base64(image: Image.Image, saliency: np.ndarray, alpha: float = 0.45) -> str:
    """Resize saliency to image size, colorize with a heatmap colormap, alpha-blend
    over the original image, and return a data: URI PNG string."""
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]

    sal = saliency.astype(np.float32)
    sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
    sal = cv2.GaussianBlur(sal, (0, 0), sigmaX=max(w, h) * 0.01)
    lo, hi = np.percentile(sal, 2), np.percentile(sal, 98)
    if hi <= lo:
        hi = lo + 1e-6
    sal = np.clip((sal - lo) / (hi - lo), 0, 1)

    heat_u8 = (sal * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    blended = (img_rgb.astype(np.float32) * (1 - alpha) + heat_color.astype(np.float32) * alpha).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
