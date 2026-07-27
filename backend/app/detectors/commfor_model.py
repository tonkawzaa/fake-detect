"""
Vendored from JeongsooP/Community-Forensics (CVPR 2025), models.py.

We vendor only the ~30-line model class rather than depending on the full
training repo (which pulls in `datasets`, DDP samplers, augmentation
pipelines that are irrelevant for inference). The class is loaded via
`PyTorchModelHubMixin.from_pretrained`, which matches state dict keys
against this exact architecture, so it must stay identical to the
upstream definition.

Source: https://github.com/JeongsooP/Community-Forensics/blob/main/models.py
"""

import torch
import torch.nn as nn
import timm
from huggingface_hub import PyTorchModelHubMixin


class ViTClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        model_size="small",
        input_size=384,
        patch_size=16,
        freeze_backbone=False,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        if model_size == "small":
            if input_size == 224:
                if patch_size == 32:
                    self.vit = timm.create_model("vit_small_patch32_224.augreg_in21k_ft_in1k", pretrained=False)
                else:
                    self.vit = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
            else:
                if patch_size == 32:
                    self.vit = timm.create_model("vit_small_patch32_384.augreg_in21k_ft_in1k", pretrained=False)
                else:
                    self.vit = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
            if freeze_backbone:
                for param in self.vit.parameters():
                    param.requires_grad = False
            self.vit.head = nn.Linear(in_features=384, out_features=1, bias=True, dtype=dtype)
        elif model_size == "tiny":
            assert patch_size == 16, "Only patch size 16 is available for ViT-Ti"
            if input_size == 224:
                self.vit = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
            else:
                self.vit = timm.create_model("vit_tiny_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
            if freeze_backbone:
                for param in self.vit.parameters():
                    param.requires_grad = False
            self.vit.head = nn.Linear(in_features=192, out_features=1, bias=True, dtype=dtype)

    def forward(self, x):
        return self.vit(x)


# imagenet normalization stats used by the upstream `dataloader.get_transform`
# test/val branch (Resize -> CenterCrop -> ToTensor[0,1] -> Normalize).
COMMFOR_NORM_MEAN = [0.485, 0.456, 0.406]
COMMFOR_NORM_STD = [0.229, 0.224, 0.225]


def commfor_resize_crop(input_size: int) -> tuple[int, int]:
    """Mirrors upstream dataloader.determine_resize_crop_sizes."""
    if input_size == 224:
        return 256, 224
    if input_size == 384:
        return 440, 384
    raise ValueError(f"Unsupported commfor input_size: {input_size}")
