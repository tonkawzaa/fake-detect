"""
Shared CLIP ViT-L/14 feature extraction.

Used identically by CLIPLinearProbeAdapter (registry.py, inference) and by
scripts/train_clip_linear_probe.py (training). Keeping the preprocessing and
pooling in one place guards against the two drifting apart -- e.g. a
different resize/crop/normalize between train and predict would silently
invalidate the trained linear weights without raising any error, since a
linear probe has no way to detect that its input distribution shifted.
"""

from __future__ import annotations

import torch
from PIL import Image

CLIP_REPO_ID = "openai/clip-vit-large-patch14"


class CLIPBackbone:
    """Frozen CLIP image encoder. Never call .train() or update its weights --
    the whole point of the linear-probe method (Ojha et al., CVPR 2023) is
    that only a linear head on top of this is trained."""

    def __init__(self, device: torch.device, repo_id: str = CLIP_REPO_ID):
        from transformers import CLIPImageProcessor, CLIPModel

        self.device = device
        self.processor = CLIPImageProcessor.from_pretrained(repo_id)
        self.model = CLIPModel.from_pretrained(repo_id)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)
        self.feature_dim = self.model.config.projection_dim

    @torch.no_grad()
    def extract(self, image: Image.Image) -> torch.Tensor:
        """Returns a single [feature_dim] feature vector for one RGB image.

        transformers>=4.5x's CLIPModel.get_image_features no longer returns a
        plain [batch, feature_dim] tensor -- it returns a
        BaseModelOutputWithPooling whose .last_hidden_state is the
        *unprojected* per-patch hidden states ([batch, n_patches+1, hidden_size])
        and whose .pooler_output has been overwritten in-place with the
        projected [batch, projection_dim] embedding (see modeling_clip.py:
        `vision_outputs.pooler_output = self.visual_projection(pooled_output)`).
        Indexing the output directly (old-API style) silently grabs
        last_hidden_state instead and breaks downstream shape assumptions.
        """
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        return self.model.get_image_features(**inputs).pooler_output[0]
