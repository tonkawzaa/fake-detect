"""
Detector adapter registry.

Each AI-detection model gets its own adapter behind a common protocol
rather than assuming every model is a plain `AutoModelForImageClassification`.
DINOv3LoRAMACAdapter and the linear-probe adapters below need custom
feature-extraction/checkpoint-loading code that doesn't fit that generic
shape; other models (SigLIP-based) use the generic `HFClassifierAdapter`.

Adapters expose predict(image) -> float in [0, 1], where 1.0 means "AI
generated". Polarity (which raw output index/sign means "fake") is a
documented footgun for these models — smoke_test_models.py asserts it
empirically rather than trusting adapter code or model card text.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DetectorAdapter(Protocol):
    name: str

    def load(self) -> None: ...

    def predict(self, image: Image.Image) -> float:
        """Return p_ai in [0, 1] for a single PIL image (RGB)."""
        ...


@dataclass
class HFClassifierAdapter:
    """Generic adapter for AutoModelForImageClassification-style HF repos."""

    repo_id: str
    name: str
    # substrings that identify the "AI/fake" class among the model's id2label
    fake_label_hints: tuple[str, ...] = ("fake", "ai", "artificial", "generated", "synthetic")

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = get_device()
        self.processor = AutoImageProcessor.from_pretrained(self.repo_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.repo_id)
        self.model.eval()
        self.model.to(self.device)

        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        fake_idx = None
        for idx, label in id2label.items():
            if any(hint in label for hint in self.fake_label_hints):
                fake_idx = idx
                break
        if fake_idx is None:
            raise ValueError(
                f"{self.repo_id}: could not find a fake/AI label in id2label={id2label}. "
                "Set fake_label_hints explicitly or inspect the model card."
            )
        self.fake_idx = fake_idx
        self.id2label = id2label

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        return probs[self.fake_idx].item()


@dataclass
class CLIPLinearProbeAdapter:
    """CLIP ViT-L/14 frozen backbone + one trained linear layer (Ojha et al.,
    "Towards Universal Fake Image Detectors that Generalize Across
    Generative Models", CVPR 2023). Their finding: because CLIP's
    contrastive pretraining never teaches it a specific generator's
    pixel-level fingerprint, a linear probe on its frozen features
    generalizes to unseen generators far better than a CNN trained
    end-to-end on the same data. Only the linear head (weight/bias) is
    trained; CLIP's own weights are never updated -- see
    scripts/train_clip_linear_probe.py for the training side and its
    honest-limitation note re: this repo's tiny training set vs. the
    paper's.

    weight/bias are a local build artifact (clip_linear_probe.json, written
    by the training script), not something downloadable from repo_id --
    load() fails loudly if it's missing rather than silently predicting
    with an untrained probe.
    """

    repo_id: str = "openai/clip-vit-large-patch14"
    name: str = "clip-vit-l14-linear-probe"
    weights_path: Path = Path(__file__).resolve().parent / "clip_linear_probe.json"

    def load(self) -> None:
        from .clip_features import CLIPBackbone

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"{self.weights_path.name} not found -- run "
                "`uv run python scripts/train_clip_linear_probe.py` to train the "
                "linear probe before this adapter can load."
            )
        probe = json.loads(self.weights_path.read_text())

        self.backbone = CLIPBackbone(device=get_device(), repo_id=self.repo_id)
        if probe["feature_dim"] != self.backbone.feature_dim:
            raise ValueError(
                f"{self.weights_path.name}: probe was trained with feature_dim="
                f"{probe['feature_dim']} but {self.repo_id} produces "
                f"{self.backbone.feature_dim}-dim features -- retrain the probe "
                "against this checkpoint."
            )
        self.weight = torch.tensor(probe["weight"], dtype=torch.float32, device=self.backbone.device)
        self.bias = float(probe["bias"])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        features = self.backbone.extract(image)
        logit = torch.dot(features, self.weight).item() + self.bias
        return 1.0 / (1.0 + math.exp(-logit))


@dataclass
class DINOv3LinearProbeAdapter:
    """Frozen DINOv3 ViT-L/16 backbone + one trained linear layer, same
    frozen-backbone-plus-linear-head shape as CLIPLinearProbeAdapter above.
    Originally proposed and measured as a cheaper alternative (~300M params)
    to the SigLIP2-giant probe (~1.9B) that used to sit in this ensemble
    slot -- it matched or exceeded SigLIP2-giant's measured accuracy at
    ~5.8x the inference speed, so SigLIP2GiantLinearProbeAdapter and its
    supporting modules (siglip2_features.py, train_siglip2_probe.py,
    perturbations.py) were removed rather than kept alongside this one; see
    the ENSEMBLE_MODEL_NAMES comment below for the swap decision and
    evaluate.py numbers. See app/detectors/dinov3_features.py for the
    backbone/checkpoint-source details (timm's ungated mirror, not the
    gated facebook/dinov3-* repo) and the feature choice (CLS token ++
    mean-pooled patch tokens -- DINOv2/DINOv3's own published linear-probe
    convention, arrived at independently of what the removed SigLIP2 probe
    used).

    Same "weights are a local build artifact, load() fails loudly if
    missing" contract as CLIPLinearProbeAdapter above.
    """

    model_name: str = "vit_large_patch16_dinov3_qkvb.lvd1689m"
    name: str = "dinov3-vit-l16-linear-probe"
    weights_path: Path = Path(__file__).resolve().parent / "dinov3_linear_probe.json"

    def load(self) -> None:
        from .dinov3_features import DINOv3Backbone

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"{self.weights_path.name} not found -- run "
                "`uv run python scripts/train_dinov3_probe.py` to train the "
                "linear probe before this adapter can load."
            )
        probe = json.loads(self.weights_path.read_text())

        self.backbone = DINOv3Backbone(device=get_device(), model_name=self.model_name)
        if probe["feature_dim"] != self.backbone.feature_dim:
            raise ValueError(
                f"{self.weights_path.name}: probe was trained with feature_dim="
                f"{probe['feature_dim']} but {self.model_name} produces "
                f"{self.backbone.feature_dim}-dim features -- retrain the probe "
                "against this checkpoint."
            )
        self.weight = torch.tensor(probe["weight"], dtype=torch.float32, device=self.backbone.device)
        self.bias = float(probe["bias"])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        features = self.backbone.extract(image)
        logit = torch.dot(features, self.weight).item() + self.bias
        return 1.0 / (1.0 + math.exp(-logit))


@dataclass
class DINOv3LoRAMACAdapter:
    """DINOv3 ViT-L/16 backbone + LoRA adapters + MAC (Multi-Attribute/
    Auxiliary Classifier) head -- main branch: real/fake (the only thing
    predict() exposes); auxiliary branch: generator-family, a training-only
    regularizer signal (see app/detectors/generator_family.py) that never
    surfaces through this adapter's interface. Unlike
    DINOv3LinearProbeAdapter above, the backbone here is NOT frozen -- LoRA
    adapters on top of it are fine-tuned end-to-end with the MAC head. See
    app/detectors/dinov3_lora_mac.py's module docstring for the model
    architecture, including an empirically-discovered footgun in this
    checkpoint's attention implementation (EvaAttention's
    qkv_bias_separate flag) that would otherwise have made the attn.qkv
    LoRA adapter silently inert.

    Trained in two stages (scripts/train_dinov3_lora_mac_stage1.py, then
    scripts/train_dinov3_lora_mac_stage2.py's self-distillation pass for
    perturbation robustness) -- this adapter loads ONLY stage 2's final
    output (dinov3_lora_mac.safetensors + _meta.json). Stage 1's own
    checkpoint (dinov3_lora_mac_stage1.safetensors) is a training-internal
    teacher snapshot consumed by stage 2 and never read here.

    Artifact shape differs from CLIPLinearProbeAdapter/
    DINOv3LinearProbeAdapter's single {weight, bias} JSON: a LoRA adapter
    plus a multi-branch head doesn't fit that shape, so this uses
    safetensors (LoRA delta weights + both head state dicts) plus a JSON
    sidecar for metadata (aux_classes, LoRA config, feature_dim). Same
    "local build artifact, load() fails loudly if missing" contract as
    those two adapters otherwise.
    """

    name: str = "dinov3-vit-l16-lora-mac"
    weights_path: Path = Path(__file__).resolve().parent / "dinov3_lora_mac.safetensors"
    meta_path: Path = Path(__file__).resolve().parent / "dinov3_lora_mac_meta.json"

    def load(self) -> None:
        from .dinov3_lora_mac import DINOv3LoRAMACModel, load_checkpoint

        if not self.weights_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                f"{self.weights_path.name} / {self.meta_path.name} not found -- run "
                "`uv run python scripts/train_dinov3_lora_mac_stage1.py` then "
                "`uv run python scripts/train_dinov3_lora_mac_stage2.py` to train this model "
                "before this adapter can load."
            )
        meta = json.loads(self.meta_path.read_text())

        self.device = get_device()
        self.model = DINOv3LoRAMACModel(
            aux_classes=meta["aux_classes"],
            lora_r=meta["lora"]["r"],
            lora_alpha=meta["lora"]["alpha"],
            lora_dropout=meta["lora"]["dropout"],
        )
        if meta["feature_dim"] != self.model.feature_dim:
            raise ValueError(
                f"{self.meta_path.name}: checkpoint was trained with feature_dim="
                f"{meta['feature_dim']} but the current model produces "
                f"{self.model.feature_dim}-dim features -- retrain before using this adapter."
            )
        self.model.to(self.device)
        load_checkpoint(self.model, path=self.weights_path)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        x = self.model.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        main_logit, _aux_logit = self.model(x)
        return torch.sigmoid(main_logit).item()


def build_candidates() -> list[DetectorAdapter]:
    """All candidates to be smoke-tested. Ensemble membership is decided
    by scripts/smoke_test_models.py, not hardcoded here.

    DINOv3LoRAMACAdapter is listed FIRST deliberately, not just
    alphabetically/historically: build_ensemble()'s list order (not
    ENSEMBLE_MODEL_NAMES's tuple order) determines adapters[0], which
    pipeline.py treats as the "primary" model for heatmap saliency (see
    app/detectors/heatmap.py). Putting the new model first is what actually
    makes it "the main model" the way it's used, not just its position in
    ENSEMBLE_MODEL_NAMES below.
    """
    return [
        DINOv3LoRAMACAdapter(),
        HFClassifierAdapter(
            repo_id="prithivMLmods/deepfake-detector-model-v1",
            name="prithivmlmods-siglip-deepfake-v1",
        ),
        HFClassifierAdapter(
            repo_id="Ateeqq/ai-vs-human-image-detector",
            name="ateeqq-ai-vs-human",
        ),
        HFClassifierAdapter(
            repo_id="Bombek1/ai-image-detector-siglip-dinov2",
            name="bombek1-siglip-dinov2",
        ),
        CLIPLinearProbeAdapter(),
        DINOv3LinearProbeAdapter(),
    ]


# Phase 0 smoke-test results (scripts/smoke_test_models.py, 25 known-real +
# 25 known-AI StyleGAN faces from TheKernel01/140k-Real-and-Fake-Faces):
#
#   community-forensics-384          polarity PASS  acc 0.84  -> REMOVED 2026-08-06, see below
#   prithivmlmods-siglip-deepfake-v1 polarity PASS  acc 0.88  -> KEPT
#   ateeqq-ai-vs-human               polarity PASS  acc 0.54  -> DROPPED (chance-level; all scores clustered near 0)
#   bombek1-siglip-dinov2            load failed (no preprocessor_config.json for AutoImageProcessor) -> DROPPED
#   clip-vit-l14-linear-probe        polarity PASS  acc 0.84  -> present in build_candidates(),
#                                     NOT in the active ensemble (see below) -- probe trained by
#                                     scripts/train_clip_linear_probe.py: out-of-fold acc
#                                     0.910, AUC 0.960 on data/clip_train/, n=310 --
#                                     see clip_linear_probe.json for the full breakdown
#   siglip2-giant-linear-probe       REMOVED 2026-07-30 -- was polarity PASS, acc 0.94, KEPT in
#                                     ENSEMBLE_MODEL_NAMES for a time (probe trained by a since-
#                                     deleted scripts/train_siglip2_probe.py: out-of-fold acc
#                                     0.903, AUC 0.949 on data/clip_train/, n=310, perturbation-
#                                     augmented). At ~1.5s/image inference -- by far the slowest
#                                     candidate ever in this ensemble (~1.9B-param vision tower,
#                                     ~3s added to every /analyze request) -- it was superseded by
#                                     dinov3-vit-l16-linear-probe below (comparable-or-better
#                                     measured accuracy at ~5.8x the speed), and its adapter,
#                                     feature module (siglip2_features.py), training script, and
#                                     supporting perturbations.py were all deleted rather than
#                                     kept as dead weight alongside the replacement.
#   dinov3-vit-l16-linear-probe      polarity PASS  acc 0.98  -> superseded, kept as fallback
#                                     (probe trained by scripts/train_dinov3_probe.py:
#                                     out-of-fold acc 0.984, AUC 0.999 on the pre-diversification
#                                     data/clip_train/, n=2012, no perturbation augmentation).
#                                     Best measured accuracy of any single candidate on both the
#                                     smoke set and data/clip_train/'s OOF numbers, at ~223ms/image,
#                                     until dinov3-vit-l16-lora-mac below. NOT deleted (unlike the
#                                     siglip2 probe above) -- kept in build_candidates() as an
#                                     explicit fallback in case the LoRA+MAC model underperforms on
#                                     the diverse-generator slice once evaluate.py is re-run.
#   dinov3-vit-l16-lora-mac          2026-08-01: polarity PASS, acc 0.980 on the smoke set
#                                     (~266ms/image) -> KEPT, PROMOTED to primary/anchor.
#                                     LoRA-fine-tuned (not frozen) DINOv3 ViT-L/16 + MAC
#                                     (Multi-Attribute/Auxiliary Classifier) head -- see
#                                     app/detectors/dinov3_lora_mac.py and
#                                     scripts/train_dinov3_lora_mac_stage1.py / _stage2.py
#                                     (self-distillation for perturbation robustness).
#                                     Stage 1 (trained on the diversified data/clip_train/,
#                                     n=2701, 8 distinct AI generator families after
#                                     de-duplication against data/eval/): held-out accuracy
#                                     0.991, AUC 1.000 (n=541); diverse-generator slice 0.977
#                                     (n=88, n_diverse_ai=71). Stage 2's self-distillation kept
#                                     clean-image accuracy identical (0.991) while raising
#                                     accuracy on a fixed perturbed (JPEG/blur/noise/resize)
#                                     copy of the held-out set from 0.837 (stage-1 teacher) to
#                                     0.972 (stage-2 student) -- confirms the stage bought real
#                                     robustness, not just a no-op refinement.
#                                     scripts/evaluate.py re-run (this pairing, on data/eval/,
#                                     n=495): accuracy 0.996, AUC 1.000 -- beats the incumbent
#                                     pairing's 0.984/0.999 on every single per-generator
#                                     breakdown (no regression anywhere), and this model's own
#                                     per-model AUC (0.999) dramatically outperforms the frozen
#                                     probe it replaced (0.610 on this same diversified set --
#                                     see dinov3-vit-l16-linear-probe's entry above, exactly the
#                                     FFHQ/StyleGAN-overfitting failure this restructuring was
#                                     meant to fix). Promoted to primary/anchor (see
#                                     build_candidates()'s ordering note) per explicit request,
#                                     while KEEPING community-forensics-384 in the ensemble
#                                     rather than replacing it outright -- it remains the only
#                                     member pretrained on broad external data, a deliberate
#                                     safety net against data/clip_train/ still being comparatively
#                                     narrow (n=2701) next to the pretraining corpora behind the
#                                     other ensemble members.
#
# This is the *default* ensemble read by ai_detector.py. Re-run the smoke
# test and update this list if models are added/removed/updated.
#
# Promotion bar that was applied above for dinov3-vit-l16-lora-mac replacing
# dinov3-vit-l16-linear-probe in ENSEMBLE_MODEL_NAMES: the challenger's
# diverse-slice (non-ffhq/stylegan) eval accuracy must be >= the incumbent
# pairing's, both measured on the same diversified data/eval/, decisive once
# n_diverse_ai >= 30. Cleared (n_diverse_ai=145, both pairings scored 1.000 on
# the diverse-AI slice specifically, and the challenger additionally improved
# on every other per-generator breakdown). dinov3-vit-l16-linear-probe is kept
# in build_candidates() as a cheap-to-revert-to fallback if a future retrain
# regresses this.
#
# ENSEMBLE_MODEL_NAMES was deliberately narrowed to two models (was:
# community-forensics-384 + prithivmlmods-siglip-deepfake-v1 +
# clip-vit-l14-linear-probe) -- community-forensics-384 as the anchor/
# baseline (highest measured AUC of any single model so far, 0.995 on
# data/eval/), paired with a second model chosen deliberately rather than
# accreting every passing candidate. prithivmlmods and the CLIP probe still
# pass the smoke test and remain valid build_candidates() entries if this
# is revisited.
#
# 2026-07-30: the second slot was siglip2-giant-linear-probe, then swapped
# for dinov3-vit-l16-linear-probe on an explicit request to test the
# pairing -- DINOv3 measured higher smoke-set accuracy (0.98 vs 0.94) at
# ~5.8x the inference speed. Re-run of scripts/evaluate.py after the swap:
# accuracy 0.986, AUC 1.000 on data/eval/ (n=207) -- down ~1 point from the
# siglip2 pairing's 0.995/1.000 (one misclassified stable_cascade sample,
# n=1, not statistically meaningful on its own), for roughly a 5-6x
# reduction in per-request latency. Kept per explicit follow-up request,
# with siglip2-giant-linear-probe's adapter and supporting files removed
# rather than left as unused dead code -- data/clip_train/'s heavy
# ffhq/stylegan skew means DINOv3's probe-training accuracy doesn't yet
# prove it generalizes as well as that comparison would suggest on its own.
#
# 2026-08-01: the anchor slot was community-forensics-384 (unchanged) paired
# with dinov3-vit-l16-linear-probe; restructured so dinov3-vit-l16-lora-mac
# (LoRA-fine-tuned, not frozen, plus a MAC head) takes the second slot AND
# is listed first in build_candidates() (see that function's ordering note)
# so it -- not community-forensics-384 -- is treated as the "main" model for
# heatmap saliency, per explicit request. See the promotion bar above.
#
# 2026-08-06: community-forensics-384 removed from the project entirely, per
# explicit request. It was kept through the 2026-08-01 restructuring as a
# safety net pretrained on broad external data, but dinov3-vit-l16-lora-mac's
# own measured accuracy (0.996/1.000 on data/eval/, see above) no longer
# needed that safety net to justify the ensemble. ENSEMBLE_MODEL_NAMES is now
# a single model. Its adapter (CommForAdapter) and the vendored model class
# it depended on (app/detectors/commfor_model.py) were deleted rather than
# left as unused code -- same discipline as the siglip2-giant-linear-probe
# removal on 2026-07-30 above. Unlike that removal, there is no fallback slot
# in build_candidates() to revert to: reintroducing this model would mean
# re-vendoring commfor_model.py from JeongsooP/Community-Forensics. Ensemble
# weights/Platt scaling were refit with scripts/evaluate.py for the
# single-model ensemble; see calibration.json.
ENSEMBLE_MODEL_NAMES: tuple[str, ...] = ("dinov3-vit-l16-lora-mac",)


def build_ensemble() -> list[DetectorAdapter]:
    """The subset of build_candidates() that passed the Phase 0 smoke test."""
    return [c for c in build_candidates() if c.name in ENSEMBLE_MODEL_NAMES]
