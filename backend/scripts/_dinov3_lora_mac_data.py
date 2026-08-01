"""
Shared manifest-loading, aux-bucketing, and train/held-out split logic for
scripts/train_dinov3_lora_mac_stage1.py and
scripts/train_dinov3_lora_mac_stage2.py.

This repo's other training scripts (train_dinov3_probe.py,
train_clip_linear_probe.py) each keep their own copy of similar-looking
helpers (load_manifest, assert_no_overlap_with_eval_set) rather than sharing
a module -- they're independent one-shot probes with no state that needs to
agree between them. Stage 1 and stage 2 here are different: stage 2 loads
stage 1's checkpoint as a teacher and MUST reproduce the exact same
train/held-out split and the exact same aux-class-name-to-index mapping, or
the student would be evaluated on images the teacher trained on, and the
loaded aux_head weights would silently correspond to the wrong classes. That
is a correctness risk, not a style preference, so this is a deliberate
exception to this repo's usual "duplicate rather than share" convention for
training scripts -- shared code here is the only way to guarantee agreement.
`build_split()` is fully deterministic given an unchanged manifest.csv (same
SEED, same AUX_CLASS_FLOOR, same row order), which is what makes reuse safe.
Leading underscore in the filename signals this is an internal helper
module, not itself a `uv run python scripts/...` entry point.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from app.detectors.generator_family import bucket_generator_family, collapse_rare_classes

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "clip_train" / "manifest.csv"
EVAL_MANIFEST_PATH = ROOT / "data" / "eval" / "manifest.csv"

CORE_GENERATORS = {"ffhq", "stylegan"}
AUX_CLASS_FLOOR = 20  # generator-family classes below this count get collapsed into other-ai
SEED = 11997733  # same seed as this repo's other calibration scripts, for consistency, not because it's shared state


def load_manifest(path: Path) -> list[tuple[Path, str, int, str]]:
    """Returns (image_path, label_str, label_int, generator) rows.
    label_str is manifest.csv's own "real"/"ai" convention (needed by
    bucket_generator_family); label_int is 1 for ai, 0 for real."""
    import csv

    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            label_str = r["label"]
            label_int = 1 if label_str == "ai" else 0
            rows.append((ROOT / r["path"], label_str, label_int, r["generator"]))
    return rows


def assert_no_overlap_with_eval_set(train_rows: list[tuple[Path, str, int, str]]) -> None:
    """Same leakage guard as train_dinov3_probe.py/train_clip_linear_probe.py
    -- re-hashes both sets rather than trusting the manifests were built
    correctly, since a silent path mistake here would invalidate any later
    comparison against evaluate.py's eval-set accuracy."""
    if not EVAL_MANIFEST_PATH.exists():
        return
    eval_rows = load_manifest(EVAL_MANIFEST_PATH)
    eval_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p, _, _, _ in eval_rows if p.exists()}
    train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p, _, _, _ in train_rows if p.exists()}
    overlap = eval_hashes & train_hashes
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} image(s) appear in BOTH {MANIFEST_PATH} and {EVAL_MANIFEST_PATH} -- "
            "training on these would leak into evaluate.py's held-out accuracy claim. "
            "Fix data/clip_train/ (re-run scripts/fetch_clip_train_set.py) before proceeding."
        )


def resolve_aux_classes(rows: list[tuple[Path, str, int, str]]) -> tuple[list[str], list[str | None]]:
    """Buckets every row's generator into a family (or None for real rows),
    collapses rare classes, and returns (surviving_aux_classes,
    per_row_final_family). Caller must check len(surviving_aux_classes) < 2
    and disable the aux branch (aux_classes=[]) if so."""
    raw_families = [bucket_generator_family(label_str, generator) for _, label_str, _, generator in rows]
    remap = collapse_rare_classes(raw_families, floor=AUX_CLASS_FLOOR)
    final_families = [remap.get(f, f) if f is not None else None for f in raw_families]

    counts = Counter(f for f in final_families if f is not None)
    survivors = sorted(counts)
    print(f"Generator-family histogram after collapse_rare_classes(floor={AUX_CLASS_FLOOR}): {dict(counts)}")
    if len(survivors) < 2:
        print(
            f"WARNING: only {len(survivors)} generator-family class(es) survive -- disabling the "
            "MAC aux branch for this run (main real/fake branch is unaffected)."
        )
        return [], final_families
    return survivors, final_families


def build_split(
    rows: list[tuple[Path, str, int, str]]
) -> tuple[list[int], list[int], list[int], list[str], list[str | None]]:
    """Returns (train_idx, heldout_idx, aux_idx_per_row, aux_classes,
    final_family_per_row). Deterministic given an unchanged manifest --
    stage 2 calls this again (not by loading saved indices) and gets the
    identical split back, which is what lets it treat stage 1's held-out
    set as still held-out."""
    aux_classes, final_families = resolve_aux_classes(rows)
    aux_class_to_idx = {c: i for i, c in enumerate(aux_classes)}
    aux_idx = [aux_class_to_idx[f] if (f is not None and f in aux_class_to_idx) else -1 for f in final_families]

    stratify_key = [f"{label_str}|{fam or 'NA'}" for (_, label_str, _, _), fam in zip(rows, final_families)]
    indices = list(range(len(rows)))
    train_idx, heldout_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=stratify_key)
    return train_idx, heldout_idx, aux_idx, aux_classes, final_families


class ManifestDataset(Dataset):
    """Loads and transforms one manifest row per __getitem__. `perturb_fn`,
    if given, is applied to the PIL image (via random_perturb) BEFORE the
    model's own transform -- used by stage 2 to train the student on
    distorted copies while stage 1 (perturb_fn=None) trains on clean
    images."""

    def __init__(
        self,
        rows: list[tuple[Path, str, int, str]],
        aux_idx: list[int],
        transform,
        perturb_fn=None,
    ):
        self.rows = rows
        self.aux_idx = aux_idx
        self.transform = transform
        self.perturb_fn = perturb_fn

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        path, _label_str, label_int, _generator = self.rows[i]
        img = Image.open(path).convert("RGB")
        if self.perturb_fn is not None:
            img = self.perturb_fn(img)
        x = self.transform(img)
        # torch.float32 explicitly -- default_collate on a bare Python float
        # produces a float64 tensor, which MPS (this repo's primary device,
        # see get_device()) cannot convert to device.
        return x, torch.tensor(float(label_int), dtype=torch.float32), self.aux_idx[i]


def masked_aux_loss(aux_logit: torch.Tensor | None, aux_idx: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over only the rows with a valid (non -1) aux label --
    ignore_index handles the masking, but F.cross_entropy divides by the
    valid count and returns nan if that count is 0 (an all-real batch), so
    that edge case is handled explicitly rather than let nan leak into the
    combined loss."""
    if aux_logit is None:
        return torch.tensor(0.0, device=aux_idx.device)
    if not bool((aux_idx != -1).any()):
        return torch.tensor(0.0, device=aux_idx.device)
    return F.cross_entropy(aux_logit, aux_idx, ignore_index=-1)


def compute_held_out_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    generators: list[str],
    aux_families: list[str | None],
    aux_classes: list[str],
) -> dict:
    preds = (probs >= 0.5).astype(int)
    overall_accuracy = float((preds == labels).mean())
    overall_auc = float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else float("nan")

    core_idx = [i for i, g in enumerate(generators) if g in CORE_GENERATORS]
    diverse_idx = [i for i, g in enumerate(generators) if g not in CORE_GENERATORS]
    core_accuracy = float((preds[core_idx] == labels[core_idx]).mean()) if core_idx else float("nan")
    diverse_accuracy = float((preds[diverse_idx] == labels[diverse_idx]).mean()) if diverse_idx else float("nan")
    n_diverse_ai = int(sum(1 for i in diverse_idx if labels[i] == 1))

    per_family_accuracy: dict[str, dict] = {}
    for cls in aux_classes:
        idx = [i for i, f in enumerate(aux_families) if f == cls]
        if idx:
            per_family_accuracy[cls] = {"n": len(idx), "accuracy": float((preds[idx] == labels[idx]).mean())}

    return {
        "overall_accuracy": overall_accuracy,
        "overall_auc": overall_auc,
        "n": len(labels),
        "core_accuracy": core_accuracy,
        "n_core": len(core_idx),
        "diverse_accuracy": diverse_accuracy,
        "n_diverse": len(diverse_idx),
        "n_diverse_ai": n_diverse_ai,
        "per_generator_family_accuracy": per_family_accuracy,
        "held_out": True,
    }
