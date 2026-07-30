"""
Shared DINOv3 ViT-L/16 feature extraction.

Used identically by DINOv3LinearProbeAdapter (registry.py, inference) and
scripts/train_dinov3_probe.py (training) -- same reason as
clip_features.py: any drift between the two would silently invalidate the
trained linear weights, since a linear probe has no way to detect that its
input distribution shifted.

Checkpoint source: timm's `vit_large_patch16_dinov3_qkvb.lvd1689m` mirror,
NOT `facebook/dinov3-vitl16-pretrain-lvd1689m` on the HF hub -- the
facebook/dinov3-* repos are gated (manual approval required), while the timm
mirrors of the same weights are ungated. `timm` is already a dependency of
this repo (see commfor_model.py), so this adds no new package. Confirmed
empirically (2026-07-30) that both point at the same lvd1689m-pretrained
ViT-L/16 weights.

Feature: CLS token concatenated with mean-pooled patch tokens (2 *
hidden_size), matching DINOv2/DINOv3's own published linear-probing
protocol (grid search over "concat CLS with avg-pooled patches or not" in
the DINOv2 paper's linear-eval recipe, which DINOv3 follows) -- there is no
equivalent "reproduce this exact paper's probe" target for DINOv3 in this
repo (unlike CLIPLinearProbeAdapter's Ojha et al. reproduction), so
concat-CLS-with-mean-patches is simply the architecture's own documented
convention, not a guess.

DINOv3 ViT prepends 4 register tokens after the CLS token, before the
patch tokens (`model.num_prefix_tokens == 5`:
1 CLS + 4 register, confirmed empirically via forward_features() output
shape and by reading timm's VisionTransformer._pos_embed, which concatenates
[cls_token, reg_token, patch_tokens] in that order). A naive
`forward_features(x).mean(dim=1)` over the whole sequence would blend
register tokens into the pooled feature -- register tokens exist
specifically to absorb global/register information the model doesn't want
attributed to any patch, so averaging them in is not a harmless no-op, it
measurably changes what the "patch mean" represents. This module always
slices prefix tokens off before pooling.
"""

from __future__ import annotations

import torch
from PIL import Image

DINOV3_MODEL_NAME = "vit_large_patch16_dinov3_qkvb.lvd1689m"


class DINOv3Backbone:
    """Frozen DINOv3 ViT-L/16 image encoder. Never call .train() or update
    its weights -- like CLIPBackbone, only a linear head on top of this is
    trained (see scripts/train_dinov3_probe.py)."""

    def __init__(self, device: torch.device, model_name: str = DINOV3_MODEL_NAME):
        import timm

        self.device = device
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)

        data_cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**data_cfg)

        self.num_prefix_tokens = self.model.num_prefix_tokens
        self.feature_dim = self.model.num_features * 2  # CLS ++ mean-pooled patches

    @torch.no_grad()
    def extract(self, image: Image.Image) -> torch.Tensor:
        """Returns a single [feature_dim] feature vector for one RGB image:
        CLS token concatenated with mean-pooled patch tokens (register
        tokens excluded from both)."""
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        tokens = self.model.forward_features(x)  # [1, num_prefix_tokens + n_patches, dim]
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, self.num_prefix_tokens :]
        pooled_patches = patch_tokens.mean(dim=1)
        return torch.cat([cls_token, pooled_patches], dim=-1)[0]
