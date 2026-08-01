"""
Stage 1 of the DINOv3-L + LoRA + MAC restructuring: trains LoRA adapters +
a two-branch MAC (Multi-Attribute/Auxiliary Classifier) head on top of a
DINOv3 ViT-L/16 backbone (app/detectors/dinov3_lora_mac.py). Main branch:
real/fake (the branch actually used at inference). Auxiliary branch:
generator-family ({"gan","diffusion","other-ai"}, see
app/detectors/generator_family.py) -- trained jointly as a regularizer, but
its loss is masked to AI-labeled rows only, so it can't just re-derive the
main label (see generator_family.py's docstring for why that mattered).

This produces dinov3_lora_mac_stage1.safetensors, a TRAINING-INTERNAL
artifact -- it is the frozen teacher checkpoint consumed by
scripts/train_dinov3_lora_mac_stage2.py's self-distillation pass, and is
never read by app/detectors/registry.py directly. The adapter registered in
registry.py (DINOv3LoRAMACAdapter) loads stage 2's output
(dinov3_lora_mac.safetensors), not this file.

Manifest loading, aux-class bucketing, and the train/held-out split are
shared with stage 2 via scripts/_dinov3_lora_mac_data.py (see that module's
docstring for why this pair of scripts is an explicit exception to this
repo's usual per-script-duplication convention).

Deliberate deviations from this repo's other training scripts' conventions
(train_dinov3_probe.py, train_clip_linear_probe.py, scripts/evaluate.py),
stated explicitly rather than silently changed:

  - Single stratified 80/20 held-out split, NOT 5-fold out-of-fold CV.
    Gradient fine-tuning a ViT-L per fold is far more expensive than an
    sklearn LogisticRegression fit per fold, and this runs on a personal Mac
    (Apple Silicon/MPS, no CUDA) -- 5x the training cost for this stage
    specifically wasn't judged worth it. Results below are reported as
    "held-out", never "out-of-fold" -- that word means something specific
    elsewhere in this repo (a stratified-CV procedure this script does not
    do) and reusing it here for a different procedure would be a quiet
    redefinition.
  - Stratified on (label, generator_family) jointly, not label alone -- a
    plain label-only 80/20 split risks putting every example of a rare
    generator family entirely on one side of the split.
  - Does NOT refit on the full dataset after reporting held-out numbers.
    Every sklearn-probe script here does that final refit because it's
    nearly free; a LoRA+MAC refit is not, and skipping it means the shipped
    weights are exactly the weights that produced the reported accuracy --
    arguably a stronger honesty property than the refit scripts have, at
    half the compute.

Usage:
    uv run python scripts/fetch_clip_train_set.py   # once, if data/clip_train/ is empty or stale
    uv run python scripts/train_dinov3_lora_mac_stage1.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from _dinov3_lora_mac_data import (
    MANIFEST_PATH,
    ManifestDataset,
    assert_no_overlap_with_eval_set,
    build_split,
    compute_held_out_metrics,
    load_manifest,
    masked_aux_loss,
)
from app.detectors.dinov3_features import DINOV3_MODEL_NAME
from app.detectors.dinov3_lora_mac import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    DINOv3LoRAMACModel,
    save_checkpoint,
)
from app.detectors.registry import get_device

ROOT = Path(__file__).resolve().parents[1]
STAGE1_WEIGHTS_PATH = ROOT / "app" / "detectors" / "dinov3_lora_mac_stage1.safetensors"
STAGE1_META_PATH = ROOT / "app" / "detectors" / "dinov3_lora_mac_stage1_meta.json"

AUX_LOSS_WEIGHT = 0.3  # hand-picked, documented as such -- same "not learned" pattern as the beauty engine's fusion weights

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8  # effective batch size 32
EVAL_BATCH_SIZE = 16
LR_LORA = 1e-4
LR_HEAD = 1e-3
EPOCHS = 3  # starting guess -- see the timing-probe print below before trusting this on a new machine
TIMING_PROBE_STEPS = 20


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found. Run scripts/fetch_clip_train_set.py first.")
        sys.exit(1)

    rows = load_manifest(MANIFEST_PATH)
    print(f"Loaded manifest: {len(rows)} images")

    print("Checking data/clip_train/ has zero overlap with data/eval/ (leakage guard)...")
    assert_no_overlap_with_eval_set(rows)
    print("  OK, no overlap")

    train_idx, heldout_idx, aux_idx, aux_classes, final_families = build_split(rows)
    print(f"Split: {len(train_idx)} train / {len(heldout_idx)} held-out (stratified on label x generator_family)")

    device = get_device()
    print(f"Device: {device}")
    model = DINOv3LoRAMACModel(aux_classes=aux_classes)
    model.to(device)
    print(
        f"Model built: {DINOV3_MODEL_NAME}, feature_dim={model.feature_dim}, "
        f"aux_classes={aux_classes}, LoRA target_modules={LORA_TARGET_MODULES}, r={LORA_R}, alpha={LORA_ALPHA}"
    )

    train_rows = [rows[i] for i in train_idx]
    train_aux = [aux_idx[i] for i in train_idx]
    heldout_rows = [rows[i] for i in heldout_idx]
    heldout_aux = [aux_idx[i] for i in heldout_idx]

    train_ds = ManifestDataset(train_rows, train_aux, model.transform)
    heldout_ds = ManifestDataset(heldout_rows, heldout_aux, model.transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    heldout_loader = DataLoader(heldout_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.main_head.parameters())
    if model.aux_head is not None:
        head_params += list(model.aux_head.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": LR_LORA}, {"params": head_params, "lr": LR_HEAD}]
    )

    print(f"\nTraining: {EPOCHS} epochs, batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS} "
          f"(effective batch {BATCH_SIZE * GRAD_ACCUM_STEPS}), lr_lora={LR_LORA}, lr_head={LR_HEAD}")

    step = 0
    timing_started = time.monotonic()
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        running_main_loss = 0.0
        running_aux_loss = 0.0
        n_batches = 0
        for batch_i, (x, label, aux_t) in enumerate(train_loader):
            x = x.to(device)
            label = label.to(device)
            aux_t = aux_t.to(device)

            main_logit, aux_logit = model(x)
            main_loss = F.binary_cross_entropy_with_logits(main_logit, label)
            aux_loss = masked_aux_loss(aux_logit, aux_t)
            loss = (main_loss + AUX_LOSS_WEIGHT * aux_loss) / GRAD_ACCUM_STEPS
            loss.backward()

            running_main_loss += main_loss.item()
            running_aux_loss += aux_loss.item()
            n_batches += 1
            step += 1

            if (batch_i + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step == TIMING_PROBE_STEPS:
                elapsed = time.monotonic() - timing_started
                sec_per_step = elapsed / TIMING_PROBE_STEPS
                steps_per_epoch = len(train_loader)
                total_steps = steps_per_epoch * EPOCHS
                projected_hours = sec_per_step * total_steps / 3600
                print(
                    f"\n[timing probe] {TIMING_PROBE_STEPS} steps took {elapsed:.1f}s "
                    f"({sec_per_step:.2f}s/step). At this rate, {total_steps} total steps "
                    f"({steps_per_epoch}/epoch x {EPOCHS} epochs) projects to ~{projected_hours:.1f}h wall-clock. "
                    "Interrupt now (Ctrl-C) and lower EPOCHS if that's not acceptable.\n"
                )

        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch {epoch + 1}/{EPOCHS}: mean main_loss={running_main_loss / n_batches:.4f} "
              f"mean aux_loss={running_aux_loss / n_batches:.4f}")

    print("\nEvaluating on held-out split...")
    model.eval()
    all_probs, all_labels, all_generators, all_families = [], [], [], []
    heldout_generators = [rows[i][3] for i in heldout_idx]
    heldout_families = [final_families[i] for i in heldout_idx]
    with torch.no_grad():
        offset = 0
        for x, label, _aux_t in heldout_loader:
            x = x.to(device)
            main_logit, _ = model(x)
            probs = torch.sigmoid(main_logit).cpu().numpy()
            bs = len(probs)
            all_probs.extend(probs.tolist())
            all_labels.extend(label.numpy().tolist())
            all_generators.extend(heldout_generators[offset : offset + bs])
            all_families.extend(heldout_families[offset : offset + bs])
            offset += bs

    metrics = compute_held_out_metrics(
        np.array(all_probs), np.array(all_labels), all_generators, all_families, aux_classes
    )
    print(f"\nHeld-out overall accuracy: {metrics['overall_accuracy']:.3f}  AUC: {metrics['overall_auc']:.3f}  n={metrics['n']}")
    print(f"  core (ffhq/stylegan) accuracy: {metrics['core_accuracy']:.3f}  n={metrics['n_core']}")
    print(f"  diverse (other generators) accuracy: {metrics['diverse_accuracy']:.3f}  n={metrics['n_diverse']}  n_diverse_ai={metrics['n_diverse_ai']}")
    for cls, d in metrics["per_generator_family_accuracy"].items():
        print(f"  aux family '{cls}': accuracy={d['accuracy']:.3f}  n={d['n']}")
    if metrics["n_diverse_ai"] < 30:
        print(
            f"\nNOTE: n_diverse_ai={metrics['n_diverse_ai']} is below the promotion bar's threshold (30) -- "
            "the diverse-slice number above is informative but not yet decisive for an incumbent-vs-challenger "
            "comparison. See scripts/evaluate.py's re-run for the actual promotion decision against data/eval/."
        )

    tensors = save_checkpoint(model, STAGE1_WEIGHTS_PATH)
    meta = {
        "model_name": DINOV3_MODEL_NAME,
        "feature_dim": model.feature_dim,
        "lora": {
            "target_modules": list(LORA_TARGET_MODULES),
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
        },
        "aux_classes": aux_classes,
        "aux_loss_weight": AUX_LOSS_WEIGHT,
        "n_train": len(train_idx),
        "epochs": EPOCHS,
        "held_out_metrics": metrics,
    }
    STAGE1_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote stage-1 checkpoint to {STAGE1_WEIGHTS_PATH} ({len(tensors)} tensors)")
    print(f"Wrote stage-1 metadata to {STAGE1_META_PATH}")
    print("\nNext: uv run python scripts/train_dinov3_lora_mac_stage2.py")


if __name__ == "__main__":
    main()
