"""
Stage 2 of the DINOv3-L + LoRA + MAC restructuring: self-distillation for
perturbation robustness ("FeatDistill" -- frozen stage-1 checkpoint as
teacher, student trained on perturbed copies of the same training images
with a feature-matching loss to the teacher's CLEAN-image features).

Why this stage exists: stage 1 (train_dinov3_lora_mac_stage1.py) trains on
clean images only. This repo used to have a perturbation-augmented training
script (train_siglip2_probe.py, since deleted alongside the SigLIP2 probe
it trained) -- app/detectors/perturbations.py was recovered from git history
specifically to give stage 2 that same robustness property back, but as a
DISTILLATION target rather than direct augmented supervision: the teacher
scores each training image once, cleanly; the student sees a randomly
perturbed (JPEG/blur/noise/resize, see perturbations.py) copy of the same
image and is trained to (a) still classify it correctly (kept losses, not
dropped) and (b) match the teacher's clean-image pooled features. If stage 2
were "same architecture, same clean images, frozen teacher" with no input
difference between teacher and student, it would be close to a no-op --
the perturbed-input requirement is what gives this stage an actual job.

Student is WARM-STARTED from stage 1's weights (not re-initialized): stage
2's objective is refining an already-decent decision boundary for
robustness, not learning representations from scratch, so starting from
stage 1's weights means the distillation loss starts small and training
time is spent on the actual point of this stage.

Efficiency: the teacher's clean-image features are precomputed ONCE per
training image before the training loop starts (mirrors
train_dinov3_probe.py's "extract features once" pattern for its frozen
backbone) rather than re-running the teacher forward pass every step --
the teacher is frozen and only ever sees clean images, so its output for a
given image never changes across epochs.

Reuses scripts/_dinov3_lora_mac_data.py for manifest loading and the
train/held-out split -- this MUST reproduce stage 1's exact split (same
seed, same manifest) or the student would be "evaluated" on images the
teacher trained on. This script asserts the recomputed aux_classes match
stage 1's saved meta and raises rather than silently proceeding if they
don't (meaning data/clip_train/ changed since stage 1 ran -- re-run stage 1
first in that case).

Produces the artifacts DINOv3LoRAMACAdapter (app/detectors/registry.py)
actually loads: app/detectors/dinov3_lora_mac.safetensors +
dinov3_lora_mac_meta.json. Stage 1's own checkpoint
(dinov3_lora_mac_stage1.safetensors) remains a training-internal
intermediate artifact, never read by registry.py.

Usage:
    uv run python scripts/train_dinov3_lora_mac_stage1.py   # once, produces the teacher checkpoint
    uv run python scripts/train_dinov3_lora_mac_stage2.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

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
from app.detectors.dinov3_lora_mac import DINOv3LoRAMACModel, load_checkpoint, save_checkpoint
from app.detectors.perturbations import random_perturb
from app.detectors.registry import get_device

ROOT = Path(__file__).resolve().parents[1]
STAGE1_WEIGHTS_PATH = ROOT / "app" / "detectors" / "dinov3_lora_mac_stage1.safetensors"
STAGE1_META_PATH = ROOT / "app" / "detectors" / "dinov3_lora_mac_stage1_meta.json"
FINAL_WEIGHTS_PATH = ROOT / "app" / "detectors" / "dinov3_lora_mac.safetensors"
FINAL_META_PATH = ROOT / "app" / "detectors" / "dinov3_lora_mac_meta.json"

DISTILL_WEIGHT = 1.0  # hand-picked -- F.mse_loss's default 'mean' reduction already
# averages over all feature_dim elements, keeping this on a comparable O(1) scale to
# the BCE/CE terms, so 1.0 is a reasonable starting weight, not a re-derived constant.

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8  # effective batch size 32, same as stage 1
EVAL_BATCH_SIZE = 16
# 2026-08-06 retune, same values and same reasoning as stage 1's (see that
# script's module docstring): LR_LORA at the bottom of a requested
# 1e-5-2e-5 range now that LORA_R quadrupled to 32, plus explicit per-group
# weight_decay (previously silently defaulted to AdamW's 0.01 on both
# groups).
LR_LORA = 1e-5
LR_HEAD = 5e-4
WD_LORA = 0.05
WD_HEAD = 0.01
EPOCHS = 1  # lowered from 2 in the same retune -- this stage warm-starts from
# an already-trained stage-1 model and its job is a robustness refinement,
# not further classification learning, so it needs even less of a nudge than
# stage 1 does. See the timing-probe print.
TIMING_PROBE_STEPS = 20
PERTURB_SEED = 11997733 + 1  # deliberately different from SEED (stage-1's split
# seed, imported from _dinov3_lora_mac_data) so perturbation randomness doesn't
# correlate with which rows landed in train vs. held-out.


class DistillDataset(Dataset):
    """Per training item: a randomly perturbed copy of the image (fresh
    perturbation draw per epoch, via a caller-owned rng) plus the teacher's
    PRECOMPUTED clean-image feature vector for that same row (looked up by
    position, not recomputed) -- see module docstring for why caching this
    once is worth doing."""

    def __init__(self, rows, aux_idx, teacher_features: torch.Tensor, transform, rng: random.Random):
        self.rows = rows
        self.aux_idx = aux_idx
        self.teacher_features = teacher_features
        self.transform = transform
        self.rng = rng

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        path, _label_str, label_int, _generator = self.rows[i]
        img = Image.open(path).convert("RGB")
        img = random_perturb(img, self.rng)
        x = self.transform(img)
        return (
            x,
            torch.tensor(float(label_int), dtype=torch.float32),
            self.aux_idx[i],
            self.teacher_features[i],
        )


def build_model_from_meta(meta: dict) -> DINOv3LoRAMACModel:
    return DINOv3LoRAMACModel(
        aux_classes=meta["aux_classes"],
        lora_r=meta["lora"]["r"],
        lora_alpha=meta["lora"]["alpha"],
        lora_dropout=meta["lora"]["dropout"],
    )


def evaluate_clean(model: DINOv3LoRAMACModel, loader: DataLoader, device, generators, families, aux_classes) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, label, _aux_t in loader:
            x = x.to(device)
            main_logit, _ = model(x)
            all_probs.extend(torch.sigmoid(main_logit).cpu().numpy().tolist())
            all_labels.extend(label.numpy().tolist())
    return compute_held_out_metrics(np.array(all_probs), np.array(all_labels), generators, families, aux_classes)


def evaluate_on_fixed_tensors(model: DINOv3LoRAMACModel, tensors: torch.Tensor, labels: list[int], device) -> float:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(tensors), EVAL_BATCH_SIZE):
            batch = tensors[start : start + EVAL_BATCH_SIZE].to(device)
            main_logit, _ = model(batch)
            preds.extend((torch.sigmoid(main_logit).cpu().numpy() >= 0.5).astype(int).tolist())
    labels_arr = np.array(labels)
    return float((np.array(preds) == labels_arr).mean())


def main() -> None:
    if not STAGE1_WEIGHTS_PATH.exists() or not STAGE1_META_PATH.exists():
        print(
            f"ERROR: {STAGE1_WEIGHTS_PATH.name} / {STAGE1_META_PATH.name} not found. "
            "Run scripts/train_dinov3_lora_mac_stage1.py first."
        )
        sys.exit(1)

    stage1_meta = json.loads(STAGE1_META_PATH.read_text())
    print(f"Loaded stage-1 metadata: aux_classes={stage1_meta['aux_classes']}, "
          f"stage-1 held-out accuracy={stage1_meta['held_out_metrics']['overall_accuracy']:.3f}")

    rows = load_manifest(MANIFEST_PATH)
    assert_no_overlap_with_eval_set(rows)
    train_idx, heldout_idx, aux_idx, aux_classes, final_families = build_split(rows)

    if sorted(aux_classes) != sorted(stage1_meta["aux_classes"]):
        raise RuntimeError(
            f"Recomputed aux_classes {sorted(aux_classes)} != stage-1's saved aux_classes "
            f"{sorted(stage1_meta['aux_classes'])} -- data/clip_train/ appears to have changed "
            "since stage 1 ran. Re-run scripts/train_dinov3_lora_mac_stage1.py before stage 2, "
            "since the aux_head's class-index mapping must match exactly what stage 1 trained."
        )

    device = get_device()
    print(f"Device: {device}")

    stage1_tensors = load_file(str(STAGE1_WEIGHTS_PATH))

    teacher = build_model_from_meta(stage1_meta).to(device)
    load_checkpoint(teacher, tensors=stage1_tensors)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = build_model_from_meta(stage1_meta).to(device)
    load_checkpoint(student, tensors=stage1_tensors)  # warm start, see module docstring

    train_rows = [rows[i] for i in train_idx]
    train_aux = [aux_idx[i] for i in train_idx]
    heldout_rows = [rows[i] for i in heldout_idx]
    heldout_aux = [aux_idx[i] for i in heldout_idx]
    heldout_generators = [rows[i][3] for i in heldout_idx]
    heldout_families = [final_families[i] for i in heldout_idx]

    print(f"\nPrecomputing teacher clean-image features for {len(train_rows)} training images...")
    clean_ds = ManifestDataset(train_rows, train_aux, teacher.transform, perturb_fn=None)
    clean_loader = DataLoader(clean_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    teacher_features = torch.zeros(len(train_rows), teacher.feature_dim)
    with torch.no_grad():
        offset = 0
        for x, _label, _aux_t in clean_loader:
            x = x.to(device)
            feats = teacher.extract_features(x).cpu()
            teacher_features[offset : offset + feats.shape[0]] = feats
            offset += feats.shape[0]
    print("  done")

    perturb_rng = random.Random(PERTURB_SEED)
    train_ds = DistillDataset(train_rows, train_aux, teacher_features, student.transform, perturb_rng)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    heldout_ds = ManifestDataset(heldout_rows, heldout_aux, student.transform)
    heldout_loader = DataLoader(heldout_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    lora_params = [p for p in student.backbone.parameters() if p.requires_grad]
    head_params = list(student.main_head.parameters())
    if student.aux_head is not None:
        head_params += list(student.aux_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": LR_LORA, "weight_decay": WD_LORA},
            {"params": head_params, "lr": LR_HEAD, "weight_decay": WD_HEAD},
        ]
    )

    aux_loss_weight = stage1_meta.get("aux_loss_weight", 0.3)
    print(f"\nTraining (self-distillation): {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
          f"grad_accum={GRAD_ACCUM_STEPS}, aux_loss_weight={aux_loss_weight}, distill_weight={DISTILL_WEIGHT}")

    step = 0
    timing_started = time.monotonic()
    for epoch in range(EPOCHS):
        student.train()
        optimizer.zero_grad()
        running = {"main": 0.0, "aux": 0.0, "distill": 0.0}
        n_batches = 0
        for batch_i, (x_pert, label, aux_t, teacher_feat) in enumerate(train_loader):
            x_pert = x_pert.to(device)
            label = label.to(device)
            aux_t = aux_t.to(device)
            teacher_feat = teacher_feat.to(device)

            student_features = student.extract_features(x_pert)
            main_logit = student.main_head(student_features).squeeze(-1)
            aux_logit = student.aux_head(student_features) if student.aux_head is not None else None

            main_loss = F.binary_cross_entropy_with_logits(main_logit, label)
            aux_loss = masked_aux_loss(aux_logit, aux_t)
            distill_loss = F.mse_loss(student_features, teacher_feat)
            loss = (main_loss + aux_loss_weight * aux_loss + DISTILL_WEIGHT * distill_loss) / GRAD_ACCUM_STEPS
            loss.backward()

            running["main"] += main_loss.item()
            running["aux"] += aux_loss.item()
            running["distill"] += distill_loss.item()
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
                    f"({sec_per_step:.2f}s/step). Projects to ~{projected_hours:.1f}h total "
                    f"({total_steps} steps). Interrupt now (Ctrl-C) and lower EPOCHS if not acceptable.\n"
                )

        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch {epoch + 1}/{EPOCHS}: mean main_loss={running['main'] / n_batches:.4f} "
              f"mean aux_loss={running['aux'] / n_batches:.4f} mean distill_loss={running['distill'] / n_batches:.4f}")

    print("\nEvaluating student on held-out split (clean images)...")
    clean_metrics = evaluate_clean(student, heldout_loader, device, heldout_generators, heldout_families, aux_classes)
    print(f"  student clean accuracy: {clean_metrics['overall_accuracy']:.3f}  AUC: {clean_metrics['overall_auc']:.3f}  n={clean_metrics['n']}")
    print(f"  (stage-1 teacher's own clean accuracy was {stage1_meta['held_out_metrics']['overall_accuracy']:.3f} -- "
          "should be close; a big drop means distillation training hurt clean-image accuracy.)")

    print("\nBuilding one fixed perturbed copy of the held-out set (same images for teacher and student)...")
    fixed_rng = random.Random(PERTURB_SEED + 1)  # yet another seed: independent of both the split and training perturbation draws
    perturbed_tensors = []
    perturbed_labels = []
    for path, _label_str, label_int, _generator in heldout_rows:
        img = Image.open(path).convert("RGB")
        img = random_perturb(img, fixed_rng)
        perturbed_tensors.append(student.transform(img))
        perturbed_labels.append(label_int)
    perturbed_tensors = torch.stack(perturbed_tensors)

    teacher_perturbed_acc = evaluate_on_fixed_tensors(teacher, perturbed_tensors, perturbed_labels, device)
    student_perturbed_acc = evaluate_on_fixed_tensors(student, perturbed_tensors, perturbed_labels, device)
    print(f"  teacher (stage-1) accuracy on perturbed held-out images: {teacher_perturbed_acc:.3f}")
    print(f"  student (stage-2) accuracy on perturbed held-out images: {student_perturbed_acc:.3f}")
    if student_perturbed_acc < teacher_perturbed_acc:
        print(
            "  NOTE: student did NOT improve on perturbed-input accuracy over the teacher -- "
            "self-distillation did not measurably buy robustness on this run. Consider more "
            "epochs, a higher DISTILL_WEIGHT, or investigate before promoting this checkpoint."
        )

    tensors_out = save_checkpoint(student, FINAL_WEIGHTS_PATH)
    meta = {
        "model_name": DINOV3_MODEL_NAME,
        "feature_dim": student.feature_dim,
        "lora": stage1_meta["lora"],
        "aux_classes": aux_classes,
        "aux_loss_weight": aux_loss_weight,
        "distill_weight": DISTILL_WEIGHT,
        "n_train": len(train_idx),
        "epochs": EPOCHS,
        "held_out_metrics": clean_metrics,
        "perturbed_accuracy": {
            "teacher_stage1": teacher_perturbed_acc,
            "student_stage2": student_perturbed_acc,
            "n": len(perturbed_labels),
        },
    }
    FINAL_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote final checkpoint to {FINAL_WEIGHTS_PATH} ({len(tensors_out)} tensors)")
    print(f"Wrote final metadata to {FINAL_META_PATH}")
    print("\nNext: uv run python scripts/smoke_test_models.py (after wiring DINOv3LoRAMACAdapter into registry.py)")


if __name__ == "__main__":
    main()
