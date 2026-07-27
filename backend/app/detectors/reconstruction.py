"""
AEROBLADE (Ricker et al., "AEROBLADE: Training-Free Detection of Latent
Diffusion Images Using Autoencoder Reconstruction Error", CVPR 2024).

Idea: a latent-diffusion model's own VAE was trained to reconstruct images
drawn from that model's own output distribution with very low error --
that's what "decode" does on every sampling step. An image genuinely
produced by that pipeline round-trips through the same VAE (encode then
decode) almost losslessly. A real photograph was never optimized for by
that VAE at all, so it round-trips with a measurably larger perceptual
error. No classifier is trained -- the reconstruction error itself is the
score; scripts/calibrate_reconstruction.py only fits the probability
mapping (1D Platt scaling) for the raw distance, the same role
scripts/evaluate.py's Platt step plays for the main ensemble.

Scope, deliberately narrow -- this is NOT a general AI-detector and is not
part of registry.py's DetectorAdapter ensemble:
  - Only detects images from the SDXL family specifically (SDXL itself,
    and distillations that keep its VAE unmodified, e.g. SSD-1B) -- see
    VAE_REPO_ID below. NOT "Stable Diffusion in general": SD1.x/2.x use a
    *different* VAE checkpoint than SDXL (shape-compatible, different
    trained weights), so an SD1.5 image has no special relationship to
    this module's VAE. An earlier version of this module used the SD1.x
    VAE (stabilityai/sd-vae-ft-mse) against SDXL-generated calibration
    data and got the polarity assertion backwards -- see
    scripts/fetch_aeroblade_calib_set.py's docstring. If you need SD1.x
    coverage, that needs its own VAE instance and its own calibration.
  - Does NOT detect GANs (StyleGAN etc. never touch this VAE at all) or
    pixel-space diffusion (ADM/DDPM/Imagen, no latent autoencoder at all)
    or other-architecture latent models (Kandinsky, Stable Cascade/
    Wuerstchen, FLUX's 16-channel VAE).

It exists to give the pipeline a second, differently-shaped opinion
specifically when the main ensemble (app/detectors/ai_detector.py, all
classifier-style models) is uncertain -- see pipeline.py, which only calls
this when ai_result.verdict == "uncertain". A classifier-style false
"uncertain" and a reconstruction-error false "uncertain" are unlikely to
fail on the same images for the same reason, so this is meant to add
signal specifically in the main ensemble's blind spot, not to replace it.

Honest limitation vs. the paper: AEROBLADE evaluates against several LDM
autoencoders and reports the *minimum* reconstruction error across all of
them, because a real-world fake could have come from any of them. This
module uses a single autoencoder for inference cost -- it will under-detect
images from LDM families with a meaningfully different autoencoder (which,
per the note above, includes plain SD1.x/2.x, not just unrelated
architectures).

CURRENT STATUS: uncalibrated, by empirical result, not by omission. With
the correct VAE (SDXL) on lossless (non-JPEG-recompressed) calibration
data, scripts/calibrate_reconstruction.py's polarity assertion still FAILS
-- and in the wrong direction by a wider margin than the initial
mismatched-VAE attempt (mean LPIPS distance: real images ~0.034, AI images
~0.061; AI should be LOWER, not higher). The only SDXL-VAE-compatible
generators available in this repo's calibration data source
(OwensLab/CommunityForensics-Eval) are LCM-LoRA distilled -- 4-8 step fast
samplers, not standard 25-50 step SDXL. The leading hypothesis is that
distillation shifts LCM-LoRA output away from the VAE decoder's typical
operating point enough to *increase* reconstruction error relative to a
clean real photo, breaking AEROBLADE's core "already on the decode
manifold" premise for this specific generator family -- but this is
unconfirmed; it hasn't been tested against non-distilled full-step SDXL
output. Until re-calibrated against such data (or investigated further),
analyze() will always return p_ai=None / verdict="uncertain" -- this is
correct, intentional behavior, not a bug to route around with a fallback
threshold.

No hand-picked fallback threshold: analyze() returns p_ai=None when
calibration.json has no "reconstruction_aeroblade" entry, rather than
guessing a probability. app/calibration.py's polarity_ok is only ever
written True by scripts/calibrate_reconstruction.py -- the script exits
before writing anything if the polarity assertion fails -- so "calibrated"
below and "polarity confirmed" are the same fact by construction. An
earlier version of this module *did* fabricate a probability from a
hand-picked DEFAULT_THRESHOLD when uncalibrated; that threshold turned out
to be wrong by a wide margin once real data was measured (real photos
landing at ~0.03 LPIPS distance, not the guessed ~0.15), which would have
confidently mislabeled real photos as AI. Don't reintroduce that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms

from .registry import get_device

# SDXL's VAE (SSD-1B is a distilled SDXL that keeps this VAE unmodified) --
# NOT stabilityai/sd-vae-ft-mse, which is SD1.x's VAE. See module docstring.
VAE_REPO_ID = "madebyollin/sdxl-vae-fp16-fix"
# SDXL is trained natively at 1024x1024; resizing both the input and its
# reconstruction to this before the LPIPS comparison keeps the metric
# in-distribution for the perceptual network too, not just the VAE.
RECON_SIZE = 1024

UNCERTAIN_MARGIN = 0.15


@dataclass
class ReconstructionResult:
    reconstruction_error: float  # raw LPIPS distance; LOWER means more likely AI (inverted vs. every other adapter)
    p_ai: float | None  # None until scripts/calibrate_reconstruction.py has confirmed polarity -- never guessed
    verdict: str  # "likely_ai" | "likely_real" | "uncertain"
    calibrated: bool
    note: str = (
        "Reconstruction-error check (AEROBLADE): only meaningful for SDXL-family "
        "latent-diffusion images, not GANs, SD1.x/2.x, or other generator architectures."
    )


class AEROBLADEDetector:
    name = "aeroblade-reconstruction"

    def load(self) -> None:
        from diffusers import AutoencoderKL
        import lpips

        self.device = get_device()

        self.vae = AutoencoderKL.from_pretrained(VAE_REPO_ID)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae.to(self.device)

        self.lpips_net = lpips.LPIPS(net="alex")
        self.lpips_net.eval()
        self.lpips_net.requires_grad_(False)
        self.lpips_net.to(self.device)

        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((RECON_SIZE, RECON_SIZE)),
                transforms.ToTensor(),  # -> [0, 1]
            ]
        )

    @torch.no_grad()
    def reconstruction_error(self, image: Image.Image) -> float:
        """Raw LPIPS distance between the image and its VAE round-trip
        reconstruction, roughly in [0, 1]. LOWER means more likely to be a
        compatible latent-diffusion image -- inverted polarity vs. every
        DetectorAdapter.predict() in registry.py, which return p_ai
        directly. Callers must not treat this as a p_ai."""
        x = self.to_tensor(image.convert("RGB")).unsqueeze(0).to(self.device)
        x_signed = x * 2.0 - 1.0  # diffusers VAEs and lpips.LPIPS both expect [-1, 1]

        latent = self.vae.encode(x_signed).latent_dist.mode()
        recon_signed = self.vae.decode(latent).sample.clamp(-1.0, 1.0)

        dist = self.lpips_net(x_signed, recon_signed)
        return float(dist.item())

    def analyze(self, image: Image.Image, calibration: dict | None = None) -> ReconstructionResult:
        distance = self.reconstruction_error(image)

        platt = (calibration or {}).get("platt")
        if platt is not None:
            logit = platt["a"] * (-distance) + platt["b"]  # inverted: lower distance -> higher p_ai
            p_ai = 1.0 / (1.0 + math.exp(-logit))
            calibrated = True
            margin = abs(p_ai - 0.5)
            if margin < UNCERTAIN_MARGIN:
                verdict = "uncertain"
            elif p_ai >= 0.5:
                verdict = "likely_ai"
            else:
                verdict = "likely_real"
        else:
            p_ai = None
            calibrated = False
            verdict = "uncertain"

        return ReconstructionResult(
            reconstruction_error=distance,
            p_ai=p_ai,
            verdict=verdict,
            calibrated=calibrated,
        )


_detector_singleton: AEROBLADEDetector | None = None


def get_reconstruction_detector() -> AEROBLADEDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = AEROBLADEDetector()
    return _detector_singleton
