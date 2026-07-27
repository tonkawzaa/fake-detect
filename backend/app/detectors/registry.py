"""
Detector adapter registry.

Each AI-detection model gets its own adapter behind a common protocol
rather than assuming every model is a plain `AutoModelForImageClassification`.
Community Forensics in particular ships a custom ViT + custom HF image
processor (see commfor_model.py) with no standard classification head to
`from_pretrained` into transformers' AutoClasses.

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
class CommForAdapter:
    """Community Forensics ViT-small/384, CVPR 2025 (JeongsooP/Community-Forensics)."""

    repo_id: str = "OwensLab/commfor-model-384"
    input_size: int = 384
    name: str = "community-forensics-384"
    # Set by smoke test after empirical verification; True means the raw
    # sigmoid(out) already equals p_ai, False means it must be flipped.
    sigmoid_is_p_ai: bool = True

    def load(self) -> None:
        from .commfor_model import ViTClassifier, commfor_resize_crop, COMMFOR_NORM_MEAN, COMMFOR_NORM_STD
        from torchvision import transforms

        self.device = get_device()
        model = ViTClassifier.from_pretrained(self.repo_id)
        model.eval()
        model.to(self.device)
        self.model = model

        resize_size, crop_size = commfor_resize_crop(self.input_size)
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=COMMFOR_NORM_MEAN, std=COMMFOR_NORM_STD),
            ]
        )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> float:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        out = self.model(x)
        p = torch.sigmoid(out).item()
        return p if self.sigmoid_is_p_ai else 1.0 - p


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
class SigLIP2GiantLinearProbeAdapter:
    """Frozen SigLIP2-giant vision tower + one trained linear layer,
    reproducing the feature side of vincentlc's NTIRE 2026 submission
    ("Robust AI-Generated Image Detection via SigLIP2-Giant and
    Perturbation-Aware Training") -- see app/detectors/siglip2_features.py
    for the exact feature (mean-pooled final-layer patch tokens, not the
    model's own pooler_output) and scripts/train_siglip2_probe.py for the
    perturbation-augmented training side, including its honest-limitation
    note re: NTIRE's official distortion pipeline being unavailable to us.

    Same "weights are a local build artifact, load() fails loudly if
    missing" contract as CLIPLinearProbeAdapter above.
    """

    repo_id: str = "google/siglip2-giant-opt-patch16-384"
    name: str = "siglip2-giant-linear-probe"
    weights_path: Path = Path(__file__).resolve().parent / "siglip2_giant_linear_probe.json"

    def load(self) -> None:
        from .siglip2_features import SigLIP2Backbone

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"{self.weights_path.name} not found -- run "
                "`uv run python scripts/train_siglip2_probe.py` to train the "
                "linear probe before this adapter can load."
            )
        probe = json.loads(self.weights_path.read_text())

        self.backbone = SigLIP2Backbone(device=get_device(), repo_id=self.repo_id)
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


def build_candidates() -> list[DetectorAdapter]:
    """All candidates to be smoke-tested. Ensemble membership is decided
    by scripts/smoke_test_models.py, not hardcoded here."""
    return [
        CommForAdapter(),
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
        SigLIP2GiantLinearProbeAdapter(),
    ]


# Phase 0 smoke-test results (scripts/smoke_test_models.py, 25 known-real +
# 25 known-AI StyleGAN faces from TheKernel01/140k-Real-and-Fake-Faces):
#
#   community-forensics-384          polarity PASS  acc 0.84  -> KEPT
#   prithivmlmods-siglip-deepfake-v1 polarity PASS  acc 0.88  -> KEPT
#   ateeqq-ai-vs-human               polarity PASS  acc 0.54  -> DROPPED (chance-level; all scores clustered near 0)
#   bombek1-siglip-dinov2            load failed (no preprocessor_config.json for AutoImageProcessor) -> DROPPED
#   clip-vit-l14-linear-probe        polarity PASS  acc 0.84  -> present in build_candidates(),
#                                     NOT in the active ensemble (see below) -- probe trained by
#                                     scripts/train_clip_linear_probe.py: out-of-fold acc
#                                     0.910, AUC 0.960 on data/clip_train/, n=310 --
#                                     see clip_linear_probe.json for the full breakdown
#   siglip2-giant-linear-probe       polarity PASS  acc 0.94  -> KEPT (probe trained by
#                                     scripts/train_siglip2_probe.py: out-of-fold acc 0.903,
#                                     AUC 0.949 on data/clip_train/, n=310, perturbation-
#                                     augmented -- see siglip2_giant_linear_probe.json.
#                                     NOTE: ~1.5s/image inference -- by far the slowest
#                                     candidate here (next is clip-vit-l14 at ~0.2s/image),
#                                     because the vision tower is ~1.9B params. ai_detector.py
#                                     calls predict() twice per request (full frame + face
#                                     crop), so this adds ~3s to every /analyze request.)
#
# This is the *default* ensemble read by ai_detector.py. Re-run the smoke
# test and update this list if models are added/removed/updated.
#
# ENSEMBLE_MODEL_NAMES was deliberately narrowed to these two models (was:
# community-forensics-384 + prithivmlmods-siglip-deepfake-v1 +
# clip-vit-l14-linear-probe) -- community-forensics-384 as the anchor/
# baseline (highest measured AUC of any single model so far, 0.995 on
# data/eval/), ensembled with siglip2-giant-linear-probe specifically to
# test that pairing per a user request, rather than accreting every
# passing candidate into one ever-growing ensemble. prithivmlmods and the
# CLIP probe still pass the smoke test and remain valid build_candidates()
# entries if this is revisited.
ENSEMBLE_MODEL_NAMES: tuple[str, ...] = (
    "community-forensics-384",
    "siglip2-giant-linear-probe",
)


def build_ensemble() -> list[DetectorAdapter]:
    """The subset of build_candidates() that passed the Phase 0 smoke test."""
    return [c for c in build_candidates() if c.name in ENSEMBLE_MODEL_NAMES]
