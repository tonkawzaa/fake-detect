# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A locally-run web app that analyses a single portrait photo and reports (1) an AI-generated-vs-real probability from a model ensemble, (2) a beauty/retouching-filter score from a hand-built pixel-statistics heuristic, and (3) EXIF/C2PA provenance. Two independent processes: a FastAPI backend (`backend/`) and a Next.js frontend (`frontend/`) that talks to it over HTTP.

## Commands

### Run everything
```bash
./run.sh   # starts backend on :8000 and frontend on :3000, Ctrl+C stops both
```

### Backend (`backend/`)
Requires Python 3.12 exactly (`requires-python = ">=3.12,<3.13"` in `pyproject.toml`) — PyTorch/MediaPipe do not ship wheels for newer Pythons, and this is a hard version pin, not a preference. Managed with `uv`.

```bash
uv sync                                              # install deps into backend/.venv
uv run uvicorn app.main:app --reload --port 8000     # dev server
uv run python scripts/smoke_test_models.py           # Phase 0 gate: verify ensemble candidates
uv run python scripts/fetch_eval_set.py              # (re)build data/eval/ from HF datasets
uv run python scripts/evaluate.py                    # k-fold calibrate AI ensemble, writes app/calibration.json
uv run python scripts/calibrate_beauty.py             # calibrate beauty thresholds, writes app/calibration.json
uv run python scripts/fetch_clip_train_set.py         # (re)build data/clip_train/ (disjoint from data/eval/) from HF datasets
uv run python scripts/train_clip_linear_probe.py      # k-fold train CLIP linear probe on data/clip_train/, writes app/detectors/clip_linear_probe.json
uv run python scripts/fetch_aeroblade_calib_set.py    # (re)build data/aeroblade_calib/ (SD-VAE-compatible generators only) from HF datasets
uv run python scripts/calibrate_reconstruction.py     # k-fold calibrate AEROBLADE reconstruction-error check, writes app/calibration.json
uv run python scripts/train_siglip2_probe.py          # k-fold train SigLIP2-giant linear probe on data/clip_train/, writes app/detectors/siglip2_giant_linear_probe.json
```

There is no test suite; correctness is checked via the scripts above (see "Calibration" below) and manual API calls, e.g. `curl -F "file=@some.jpg" localhost:8000/analyze`.

### Frontend (`frontend/`)
```bash
npm install
npm run dev      # :3000, expects backend at NEXT_PUBLIC_API_URL (.env.local, defaults to :8000)
npm run build
npm run lint
```

**This Next.js install (16.x) has breaking changes from older Next.js knowledge** — `frontend/AGENTS.md` says to check `frontend/node_modules/next/dist/docs/` before relying on prior training data for App Router/API conventions. Notably, `mediapipe`'s old `mp.solutions` API is also gone in the installed version (0.10.35) — only the Tasks API (`mediapipe.tasks.vision`) exists; see `app/detectors/face_gate.py`.

## Architecture

### Request flow (`backend/app/pipeline.py`)
`POST /analyze` → `face_gate` (MediaPipe FaceLandmarker) gates the input to "is this a usable portrait" → if not, returns `status: "no_face"|"low_quality"` with no fabricated scores → if yes, runs three independent stages and merges results, each wrapped so one stage failing doesn't 500 the whole request:
1. **AI-detection ensemble** (`app/detectors/ai_detector.py`) — runs each model in `registry.build_ensemble()` on both the full frame and a margin-padded face crop, fuses via weighted logit averaging + Platt scaling
2. **Beauty engine** (`app/beauty/`) — pixel-statistics features over FaceMesh-derived regions
3. **Provenance** (`app/forensics/`) — EXIF (`piexif`) + C2PA (`c2pa-python`)

Plus a saliency heatmap (`app/detectors/heatmap.py`) computed via input-gradient on the primary ensemble model (not attention rollout — timm's fused attention path doesn't expose per-head weights without fragile monkeypatching, so gradient saliency was used instead; see the module docstring).

**Only when the AI-detection ensemble's verdict is `"uncertain"`**, pipeline.py runs one more, differently-shaped check: `app/detectors/reconstruction.py`'s AEROBLADE reconstruction-error detector (frozen Stable-Diffusion VAE encode/decode round-trip + LPIPS distance, training-free — see its module docstring for why this is deliberately *not* another `registry.py` ensemble member and why its detection scope is narrow — SD-family latent-diffusion images only, not GANs). It's a secondary opinion surfaced as `reconstruction_check` in the response, never blended into the calibrated `ai_probability`.

### Detector adapter pattern (`app/detectors/registry.py`)
Every AI-detection model implements a small protocol (`load()`, `predict(image) -> p_ai`) rather than assuming a uniform `AutoModelForImageClassification` interface — **Community Forensics is not a standard HF classifier**: it's a custom `timm` ViT with its own checkpoint format, vendored (not pip-installed) into `app/detectors/commfor_model.py` because the upstream repo's inference path is a notebook, not a package. Other models (SigLIP-based) use the generic `HFClassifierAdapter`.

`build_candidates()` lists everything that *could* be in the ensemble; `ENSEMBLE_MODEL_NAMES` / `build_ensemble()` is the subset that actually passed `scripts/smoke_test_models.py`. **Never add a model to `ENSEMBLE_MODEL_NAMES` without running the smoke test first** — it asserts label polarity empirically (`mean(p_ai)` on known-AI images must exceed it on known-real images) because model cards and even upstream example notebooks have been observed to state the wrong polarity for their own outputs. A model can look like it loads and runs fine while silently scoring backwards.

`CLIPLinearProbeAdapter` is a third pattern alongside the vendored-ViT and generic-HF-classifier ones above: a frozen CLIP ViT-L/14 backbone (`app/detectors/clip_features.py`, shared verbatim between inference and training so the two can't drift apart) topped with one trained linear layer, reproducing Ojha et al. (CVPR 2023) — the paper's point is that a linear probe on frozen CLIP features generalizes to *unseen* generators better than an end-to-end CNN, since CLIP's contrastive pretraining doesn't memorize any one generator's fingerprint. Its weight/bias live in `app/detectors/clip_linear_probe.json`, a build artifact written by `scripts/train_clip_linear_probe.py` (same out-of-fold k-fold discipline as `scripts/evaluate.py`) — `load()` raises `FileNotFoundError` until that script has been run, and like every other candidate it still needs to clear the smoke test before joining `ENSEMBLE_MODEL_NAMES`. **It trains on `data/clip_train/` (`scripts/fetch_clip_train_set.py`), never on `data/eval/`** — `data/eval/` is what `scripts/evaluate.py` reports "measured accuracy" against, and this is an ensemble member whose weights are actually fit by this repo (as opposed to the pretrained-externally ones), so training it on the same images evaluate.py scores would leak labels into that accuracy claim by a path evaluate.py's own k-fold can't detect. `fetch_clip_train_set.py` guards the split with a content-hash check against `data/eval/`, and `train_clip_linear_probe.py` re-checks it before fitting. (`clip-vit-l14-linear-probe` currently passes the smoke test but is not in the active `ENSEMBLE_MODEL_NAMES` -- see below.)

`SigLIP2GiantLinearProbeAdapter` follows the exact same frozen-backbone-plus-linear-head shape, this time reproducing the feature side of vincentlc's NTIRE 2026 submission ("Robust AI-Generated Image Detection via SigLIP2-Giant and Perturbation-Aware Training"): `google/siglip2-giant-opt-patch16-384`'s vision tower (`app/detectors/siglip2_features.py`, ~1.9B params, vision-only — the text tower is never loaded), feature = mean pooling over the *final hidden layer's patch tokens* (deliberately not the model's own attention-pooled `pooler_output`, matching vincentlc's description). `scripts/train_siglip2_probe.py` also reuses `data/clip_train/` (same disjointness guarantee as the CLIP probe — a different model training on the same already-disjoint set doesn't reopen the leakage risk) and additionally trains on a mix of clean and perturbed image copies (`app/detectors/perturbations.py`: JPEG/blur/noise/resize distortions at multiple severities) to approximate vincentlc's "perturbation-aware training" — NTIRE's actual official distortion pipeline isn't published, so this is our own stand-in, documented as an honest limitation in that module's docstring. Because each source image yields multiple correlated feature rows (1 clean + N augmented), the k-fold split there uses `StratifiedGroupKFold` grouped by source image, not plain `StratifiedKFold` — letting augmented copies of a held-out image leak into a training fold would be the same class of bug as training on the eval set itself. **This is by far the heaviest ensemble candidate (~1.5s/image inference vs. ~50-200ms for everything else)** — `ai_detector.py` calls `predict()` twice per request (full frame + face crop), so it adds roughly 3 seconds to every `/analyze` request when active.

**`ENSEMBLE_MODEL_NAMES` is currently `(community-forensics-384, siglip2-giant-linear-probe)`** — narrowed from the earlier 3-model ensemble (`community-forensics-384` + `prithivmlmods-siglip-deepfake-v1` + `clip-vit-l14-linear-probe`) per an explicit request to pair `community-forensics-384` as baseline with the SigLIP2-giant probe specifically, not to accrete every passing candidate. `prithivmlmods-siglip-deepfake-v1` and `clip-vit-l14-linear-probe` still pass the smoke test and remain valid `build_candidates()` entries. Recalibrated via `scripts/evaluate.py`: out-of-fold accuracy 0.995, AUC 1.000 on `data/eval/` (n=207), near-equal fitted weights (~0.50/0.50) between the two models.

### Reconstruction-error fallback (`app/detectors/reconstruction.py`) — implemented, wired, currently uncalibrated by empirical result
AEROBLADE (Ricker et al., CVPR 2024): the premise is that an image genuinely produced by a given latent-diffusion model round-trips through *that model's own VAE* (encode then decode) with very low LPIPS error, because that's exactly what the VAE decoder does on every real sampling step, while a real photo — never optimized for by that VAE — round-trips with measurably more error. No classifier is trained — `scripts/calibrate_reconstruction.py` only fits a 1D Platt scaling on top of the raw distance, same role as `scripts/evaluate.py`'s Platt step for the main ensemble, and **refuses to write a calibration entry if the polarity assertion (mean error on real > mean error on AI) fails**, same discipline as `scripts/smoke_test_models.py`'s DROP outcome.

**That assertion currently fails, on the best data and VAE available to this repo.** Scope is narrow by design: only SDXL-VAE-compatible generators are targeted (`app/detectors/reconstruction.py`'s `VAE_REPO_ID = madebyollin/sdxl-vae-fp16-fix`; verified empirically against `OwensLab/CommunityForensics-Eval`'s `model_name` values that only `LCM_lora_sdxl`/`LCM_lora_ssd1b` qualify — SD1.x/2.x use a *different* VAE checkpoint despite being "Stable Diffusion" too, and look-alikes like `stable_cascade`/`kandinsky_2_2`/`FLUX-dev` use architecturally different autoencoders entirely; see `scripts/fetch_aeroblade_calib_set.py`'s docstring). A first calibration attempt (mismatched SD1.x VAE + lossy JPEG-recompressed calibration images) failed polarity backwards. Both issues were fixed — correct SDXL VAE, lossless images, native 1024px resolution — and polarity **still fails, by a wider margin** (mean LPIPS distance: real ≈0.034, AI ≈0.061; AI should be lower, not higher). The leading hypothesis is that the only SDXL-VAE-compatible generators available in this data source are LCM-LoRA distilled (4-8 step fast samplers), and that distillation shifts output away from the VAE's typical decode manifold enough to break AEROBLADE's core premise for this specific generator family — unconfirmed, since non-distilled full-step SDXL calibration data wasn't available to test against. See `app/detectors/reconstruction.py`'s module docstring for the full account.

Because of this, `AEROBLADEDetector.analyze()` has **no fallback threshold** — it deliberately returns `p_ai=None` / `verdict="uncertain"` rather than guess (an earlier version *did* guess a hand-picked default and it was wrong by a wide margin once measured, which would have confidently mislabeled real photos). It's still wired as a *conditional secondary check* in `pipeline.py` (only runs when the main ensemble's verdict is `"uncertain"`, surfaced as `reconstruction_check` with the raw uncalibrated distance) rather than a `registry.py` ensemble member, and the frontend (`VerdictCard.tsx`) shows the raw distance with "not calibrated yet" rather than a verdict phrase. Re-running `scripts/calibrate_reconstruction.py` after sourcing non-distilled SDXL calibration data is the natural next step if this is revisited.

### Beauty engine (`app/beauty/`)
`regions.py` builds skin/eye/lips/border masks from MediaPipe FaceMesh landmarks by walking the named connection sets (`FACE_LANDMARKS_FACE_OVAL` etc.) into ordered polygons — not hardcoded landmark-index lists. `features.py` computes 8 hand-designed features (F1-F8, see module docstring) comparing skin-region pixel statistics against eye/border reference regions; F5 is a guard multiplier that dampens the whole score when the reference region is itself soft (out-of-focus/low-res photo), so a blurry-but-unfiltered photo doesn't get misread as retouched. `score.py` fuses features into a single score via a hand-weighted (not learned) formula — there's no labeled beauty-filter dataset at scale, so this is a heuristic, not a classifier, and every sub-score is surfaced in the API response rather than only the fused number.

### Calibration (`app/calibration.py`, `app/calibration.json`)
`calibration.json` is a build artifact, not source — it's regenerated by `scripts/evaluate.py` (AI ensemble: per-model weights, Platt scaling params, decision threshold, accuracy/AUC/per-generator breakdown) and `scripts/calibrate_beauty.py` (beauty feature thresholds). Both scripts fit on `data/eval/` / `data/beauty_pairs/`, which are gitignored data directories, not code — read `pipeline.py` and `app/calibration.py` to see how the app degrades gracefully (uncalibrated ensemble average, hand-picked default thresholds) when calibration hasn't been run yet.

`scripts/evaluate.py` fits ensemble weights/Platt scaling with stratified k-fold CV and reports **out-of-fold** accuracy specifically to avoid reporting train-set performance as if it were held-out accuracy — don't change this to report in-sample numbers without a strong reason, it's a deliberate correctness property of the whole "measured accuracy" claim shown in the UI.

`scripts/fetch_eval_set.py` builds `data/eval/manifest.csv` (path, label, generator) from two HF dataset sources (a face-specific StyleGAN/FFHQ dataset, and the general-purpose Community Forensics eval set filtered through `face_gate` to keep only portraits) — see its docstring for the label-polarity gotcha in the second source (`label` there indicates real/generated, but `model_name` on a *real* row identifies which generator's fake it's paired with, not that the real row itself came from that model).

### API contract
`backend/app/schemas.py` is the single source of truth for the response shape; `frontend/src/lib/types.ts` mirrors it by hand (no codegen) — when changing one, update the other. The API deliberately exposes two different "percentage" concepts that must not be conflated in either backend responses or frontend copy: per-image model confidence (`ai_probability`, free, always available) vs. measured ensemble accuracy on the eval set (`model_accuracy`, requires calibration to have been run, out-of-fold).
