"""
Trainable DINOv3 ViT-L/16 backbone + LoRA adapters + MAC (Multi-Attribute/
Auxiliary Classifier) head: main branch = real/fake, auxiliary branch =
generator-family (see app/detectors/generator_family.py).

This deliberately does NOT reuse DINOv3Backbone (dinov3_features.py) even
though it wraps the identical checkpoint (DINOV3_MODEL_NAME) -- that class
hard-codes `.eval()` / `requires_grad_(False)` / `@torch.no_grad()` in its
own `__init__`/`extract()`, because every other consumer of it
(DINOv3LinearProbeAdapter, scripts/train_dinov3_probe.py) is fitting a
linear probe on FROZEN features and must never accidentally update backbone
weights. LoRA fine-tuning needs gradients flowing through (a low-rank
adapter on) those weights, which is structurally incompatible with that
class's contract. This is a deliberate, documented exception to this
codebase's usual "share verbatim between training and inference" rule (see
dinov3_features.py's own docstring for that rule) -- the part that DOES stay
shared is the pure tensor pooling step (`pool_dinov3_tokens`, imported from
dinov3_features.py, not re-derived here), since that has no relationship to
whether gradients are flowing.

LoRA target modules were verified empirically against this exact checkpoint
(`vit_large_patch16_dinov3_qkvb.lvd1689m`), not guessed:
`timm`'s fused-QKV ViT block exposes `attn.qkv` (nn.Linear, fused Q+K+V),
`attn.proj` (nn.Linear, attention output projection), `mlp.fc1`/`mlp.fc2`
(nn.Linear) -- confirmed via `named_modules()` (24 of each, one per ViT-L
block) and by checking `peft.get_peft_model()`'s actual injection count
equals `4 * len(blocks)` = 96, not just that it didn't error. This matters
because `patch_embed.proj` is a *separate* Conv2d whose name also ends in
"proj" -- PEFT's target_modules matching is suffix-based, so an unqualified
`target_modules=["proj"]` would silently also LoRA-adapt the patch
embedding. Using the dotted, qualified suffixes below (`attn.proj`, not
`proj`) avoids that; `patch_embed.proj` was confirmed to remain untouched
(still `torch.nn.Conv2d`, not LoRA-wrapped) after applying this config.

peft.get_peft_model() (task_type=None, the generic/non-causal-LM path)
returns a PeftModel that delegates unknown attribute access to the wrapped
base model -- `forward_features()`, `num_prefix_tokens`, `num_features`, and
`blocks` all remain directly callable/readable on the wrapped object, so
this module calls `self.backbone.forward_features(x)` exactly the way
DINOv3Backbone.extract() does, rather than `self.backbone(x)` (which would
route through timm's own classification head/pooling -- irrelevant here
since num_classes=0, but also not the CLS++mean-pooled-patches convention
this repo has standardized on).

IMPORTANT, empirically-discovered footgun: this checkpoint's attention
class is `timm.models.eva.EvaAttention` (the "qkvb" in the checkpoint name
means it keeps `q_bias`/`k_bias`/`v_bias` as separate learnable Parameters
rather than folding them into `qkv`'s own bias -- `attn.qkv` here is a
bias-free nn.Linear). Its forward has TWO code paths depending on
`attn.qkv_bias_separate` (confirmed by reading timm/models/eva.py's
`EvaAttention.forward`, not guessed): when False (this checkpoint's
as-loaded default, confirmed empirically), it computes
`F.linear(x, weight=self.qkv.weight, bias=qkv_bias)` -- calling `F.linear`
directly on the raw `.weight` tensor, which completely BYPASSES
`self.qkv.__call__()`/`.forward()`. Since PEFT's LoRA injection works by
wrapping a module's forward, this means a LoRA adapter placed on `attn.qkv`
would be silently inert: it loads, `get_peft_model()` reports it as
injected, but it never participates in the forward pass and its gradient
never fires (confirmed empirically: all 48 attn.qkv LoRA parameters across
24 blocks had `.grad is None` after backward, while attn.proj/mlp.fc1/fc2
LoRA parameters all received gradients normally). When
`qkv_bias_separate=True` instead, the forward takes the other branch
(`qkv = self.qkv(x); qkv += qkv_bias`) -- a genuine module call that PEFT's
hook does intercept -- and this is numerically an EXACT no-op relative to
the `False` path (verified: max abs diff 0.0 between forward_features()
outputs on real pretrained weights with vs. without this flag flipped,
since both compute the identical `x @ W^T + bias`, just fused differently).
`_build_lora_backbone()` therefore flips `qkv_bias_separate = True` on every
block's attention module BEFORE calling `get_peft_model()`, so the
`attn.qkv` LoRA adapter is real rather than dead weight sitting in the
checkpoint. This is the difference between "the injection count assertion
passed" and "the injected modules actually do anything" -- passing the
count check alone would NOT have caught this.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .dinov3_features import DINOV3_MODEL_NAME, pool_dinov3_tokens

LORA_TARGET_MODULES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
# r=32/alpha=64 (alpha = 2*r, a common default) per explicit request, matching
# ranks reported by other teams (Shallow Real, Ant International) doing LoRA
# fine-tuning of ViT backbones for this task -- trains ~1-3% of total
# parameters while approaching full-fine-tune accuracy. Was r=8/alpha=16
# before the 2026-08-06 retune; see scripts/train_dinov3_lora_mac_stage1.py's
# module docstring for the paired learning-rate/weight-decay changes and the
# reasoning (avoid overfitting a well-pretrained backbone to the generators
# seen in data/clip_train/).
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05


def _build_lora_backbone(model_name: str, r: int, alpha: int, dropout: float):
    """Loads a TRAINABLE (not frozen) DINOv3 ViT-L/16 and wraps it with
    LoRA. Returns (peft_model, raw_backbone) -- raw_backbone is kept only to
    resolve the input transform and to assert the injection count, since
    timm.data.resolve_data_config expects the plain timm model, not the
    peft-wrapped one."""
    import timm
    from peft import LoraConfig, get_peft_model

    raw_backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    data_cfg = timm.data.resolve_data_config({}, model=raw_backbone)
    transform = timm.data.create_transform(**data_cfg)

    # See module docstring: this checkpoint's EvaAttention defaults to
    # qkv_bias_separate=False, which bypasses self.qkv's module call (and
    # thus any LoRA wrapping of it) via a direct F.linear on .weight.
    # Flipping this is a numerically-exact no-op that makes attn.qkv's LoRA
    # adapter real instead of dead weight. Must happen before
    # get_peft_model() wraps attn.qkv, since the flag is read on every
    # forward call, not just once.
    for block in raw_backbone.blocks:
        block.attn.qkv_bias_separate = True

    lora_config = LoraConfig(
        target_modules=list(LORA_TARGET_MODULES),
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
    )
    peft_backbone = get_peft_model(raw_backbone, lora_config)

    injected = sum(1 for _, mod in peft_backbone.named_modules() if hasattr(mod, "lora_A"))
    expected = len(LORA_TARGET_MODULES) * len(raw_backbone.blocks)
    if injected != expected:
        raise RuntimeError(
            f"Expected LoRA to inject into {expected} modules "
            f"({len(LORA_TARGET_MODULES)} target types x {len(raw_backbone.blocks)} blocks), "
            f"got {injected}. target_modules={LORA_TARGET_MODULES!r} may not match this "
            f"checkpoint's module names -- re-run `named_modules()` and check before trusting "
            "this model."
        )

    return peft_backbone, transform


class DINOv3LoRAMACModel(nn.Module):
    """Trainable DINOv3 ViT-L/16 (LoRA-adapted) + MAC head.

    `aux_classes`: the ordered list of generator-family class names this
    model's aux_head predicts (see app/detectors/generator_family.py). Pass
    an empty list/tuple to disable the aux branch entirely (e.g. if
    surviving_aux_classes() found fewer than 2 viable classes on the
    training data) -- `aux_head` is then None and `forward()`'s second
    return value is always None.

    Never call `.eval()` / `requires_grad_(False)` on this module or its
    `.backbone` from outside training code -- unlike DINOv3Backbone, this
    class's entire purpose is to have its LoRA parameters (and both heads)
    updated by gradient descent.
    """

    def __init__(
        self,
        aux_classes: tuple[str, ...] | list[str],
        model_name: str = DINOV3_MODEL_NAME,
        lora_r: int = LORA_R,
        lora_alpha: int = LORA_ALPHA,
        lora_dropout: float = LORA_DROPOUT,
    ):
        super().__init__()
        self.backbone, self.transform = _build_lora_backbone(model_name, lora_r, lora_alpha, lora_dropout)
        self.num_prefix_tokens = self.backbone.num_prefix_tokens
        self.feature_dim = self.backbone.num_features * 2  # CLS ++ mean-pooled patches

        self.aux_classes = list(aux_classes)
        self.main_head = nn.Linear(self.feature_dim, 1)
        self.aux_head = nn.Linear(self.feature_dim, len(self.aux_classes)) if self.aux_classes else None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: preprocessed image batch [B, 3, H, W]. Returns [B, feature_dim]."""
        tokens = self.backbone.forward_features(x)  # [B, num_prefix_tokens + n_patches, dim]
        return pool_dinov3_tokens(tokens, self.num_prefix_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (main_logit [B], aux_logit [B, n_aux_classes] or None)."""
        features = self.extract_features(x)
        main_logit = self.main_head(features).squeeze(-1)
        aux_logit = self.aux_head(features) if self.aux_head is not None else None
        return main_logit, aux_logit


def save_checkpoint(model: DINOv3LoRAMACModel, path: Path) -> dict[str, torch.Tensor]:
    """Saves LoRA delta weights (not the frozen base backbone) + both head
    state dicts into ONE safetensors file -- deliberately not peft's own
    save_pretrained() directory format, to match this repo's single-file
    artifact convention (clip_linear_probe.json, dinov3_linear_probe.json).
    Returns the exact tensor dict that was written, so callers (stage-2
    training) can reuse it in-memory as a teacher/warm-start without a
    round-trip through disk."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    for k, v in get_peft_model_state_dict(model.backbone).items():
        tensors[f"lora.{k}"] = v.detach().contiguous().cpu()
    for k, v in model.main_head.state_dict().items():
        tensors[f"main_head.{k}"] = v.detach().contiguous().cpu()
    if model.aux_head is not None:
        for k, v in model.aux_head.state_dict().items():
            tensors[f"aux_head.{k}"] = v.detach().contiguous().cpu()

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return tensors


def load_checkpoint(model: DINOv3LoRAMACModel, path: Path | None = None, tensors: dict[str, torch.Tensor] | None = None) -> None:
    """Loads a checkpoint produced by save_checkpoint() into `model` in
    place. Pass exactly one of `path` (read from disk) or `tensors` (reuse
    an in-memory dict, e.g. stage-2 loading its own just-saved stage-1
    teacher without a disk round-trip)."""
    if (path is None) == (tensors is None):
        raise ValueError("load_checkpoint: pass exactly one of path or tensors")

    if tensors is None:
        from safetensors.torch import load_file

        tensors = load_file(str(path))

    from peft import set_peft_model_state_dict

    lora_sd = {k[len("lora.") :]: v for k, v in tensors.items() if k.startswith("lora.")}
    main_sd = {k[len("main_head.") :]: v for k, v in tensors.items() if k.startswith("main_head.")}
    aux_sd = {k[len("aux_head.") :]: v for k, v in tensors.items() if k.startswith("aux_head.")}

    set_peft_model_state_dict(model.backbone, lora_sd)
    model.main_head.load_state_dict(main_sd)
    if aux_sd:
        if model.aux_head is None:
            raise ValueError(
                f"Checkpoint {path if path else '(in-memory)'} has aux_head weights but this "
                "model was built with aux_classes=[] -- rebuild it with the same aux_classes "
                "the checkpoint was trained with."
            )
        model.aux_head.load_state_dict(aux_sd)
