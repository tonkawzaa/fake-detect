"""
Shared SigLIP2-giant vision-tower feature extraction.

Used identically by SigLIP2GiantLinearProbeAdapter (registry.py, inference)
and scripts/train_siglip2_probe.py (training) -- same reason as
clip_features.py: any drift between the two would silently invalidate the
trained linear weights, since a linear probe has no way to detect that its
input distribution shifted.

Reproduces the feature side of vincentlc's NTIRE 2026 submission ("Robust
AI-Generated Image Detection via SigLIP2-Giant and Perturbation-Aware
Training"): backbone is google/siglip2-giant-opt-patch16-384, and the
feature is mean pooling over the patch-token sequence from the encoder's
final hidden layer -- NOT the model's own pooler_output (SiglipVisionModel
computes that via a learned multi-head attention pooling head, MAP), which
vincentlc's method deliberately bypasses in favor of a plain average. Using
pooler_output instead would silently change what the linear probe is
trained on vs. what a from-memory reproduction might reach for.

Only the vision tower is loaded (SiglipVisionModel, not the full
SiglipModel) -- this is a vision-only linear probe with no use for the text
tower, and skipping it avoids instantiating/holding ~700M extra text-model
parameters. Note this does NOT save download bandwidth: the checkpoint
ships as sharded safetensors mixing vision_model.* and text_model.* keys,
so `from_pretrained` still fetches the full ~1.9B-parameter checkpoint
before discarding the unused text_model.* weights.
"""

from __future__ import annotations

import torch
from PIL import Image

SIGLIP2_REPO_ID = "google/siglip2-giant-opt-patch16-384"


class SigLIP2Backbone:
    """Frozen SigLIP2-giant vision tower. Never call .train() or update its
    weights -- like CLIPBackbone, only a linear head on top of this is
    trained (see app/detectors/registry.py: SigLIP2GiantLinearProbeAdapter)."""

    def __init__(self, device: torch.device, repo_id: str = SIGLIP2_REPO_ID):
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(repo_id)
        self.model = SiglipVisionModel.from_pretrained(repo_id)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        self.feature_dim = self.model.config.hidden_size

    @torch.no_grad()
    def extract(self, image: Image.Image) -> torch.Tensor:
        """Returns a single [feature_dim] feature vector for one RGB image:
        mean-pooled patch tokens from the final hidden layer."""
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        patch_tokens = self.model(**inputs).last_hidden_state  # [1, n_patches, feature_dim]
        return patch_tokens.mean(dim=1)[0]
