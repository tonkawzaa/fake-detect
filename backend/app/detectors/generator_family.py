"""
Generator-family bucketing for the MAC (Multi-Attribute/Auxiliary Classifier)
head's auxiliary branch (see scripts/train_dinov3_lora_mac_stage1.py).

The auxiliary branch's job is to predict *which family of generator* produced
a fake image -- a signal genuinely different from the main real/fake branch,
not a relabeling of it. Two design choices make that true rather than
accidental:

1. Real-labeled rows get no aux label at all (`bucket_generator_family`
   returns None for them) -- the aux loss is masked to AI-labeled rows only
   at training time. This also makes a real footgun in this repo's data
   harmless by construction: for CommunityForensics-Eval rows, the manifest's
   `generator` column means different things depending on `label` -- for
   AI rows it's the actual generator (`model_name`), but for REAL rows it's
   the *paired dataset name* (`real_source`, e.g. "LAION"), not a generator
   at all (see fetch_clip_train_set.py / fetch_eval_set.py docstrings). Since
   real rows never reach this function's AI branch, that footgun can't
   silently poison the aux head with nonsense classes like "LAION".

2. Individual generator names are collapsed into a small set of coarse
   families ({"gan", "diffusion", "other-ai"}) rather than kept as
   per-generator classes. This repo's training data (data/clip_train/,
   data/eval/) has, even after diversification, wildly uneven per-generator
   counts (e.g. thousands of StyleGAN rows vs. single-digit counts for any
   one diffusion checkpoint) -- a per-generator-name class scheme would give
   many classes ~1-2 samples, which breaks stratified splits and teaches
   nothing. Family-level buckets are what the sample counts can actually
   support.

The keyword lists below are informed by generator names already verified
empirically elsewhere in this repo (scripts/fetch_aeroblade_calib_set.py's
model_name scan of OwensLab/CommunityForensics-Eval: LCM_lora_sdxl,
LCM_lora_ssd1b, LCM_lora_sdv15, stable_cascade, kandinsky_2_2, FLUX-dev; this
repo's own StyleGAN/DFGAN sources) plus well-known commercial/open generator
names likely to appear once data/clip_train/'s CF-Eval slice is scanned more
broadly (MidjourneyV6_1, DALL-E, Imagen, Firefly). This is a hand-maintained,
documented mapping, hand-picked rather than learned -- not an attempt at a
complete taxonomy. Anything not matched falls into "other-ai" by design,
so an unrecognized generator name from a freshly-fetched manifest degrades
to a coarser-than-ideal bucket rather than crashing training.
"""

from __future__ import annotations

from collections import Counter

_GAN_KEYWORDS = (
    "stylegan",
    "dfgan",
    "biggan",
    "cyclegan",
    "stargan",
    "progan",
    "gan",  # broad catch-all; checked after the more specific names above
)

_DIFFUSION_KEYWORDS = (
    "lcm",
    "sdxl",
    "sdv15",
    "sd15",
    "sd_1_5",
    "sd-1-5",
    "stable",
    "cascade",
    "kandinsky",
    "flux",
    "midjourney",
    "dalle",
    "dall-e",
    "imagen",
    "firefly",
    "glide",
    "diffusion",
)

OTHER_AI = "other-ai"
GAN = "gan"
DIFFUSION = "diffusion"

AUX_CLASSES = (GAN, DIFFUSION, OTHER_AI)


def bucket_generator_family(label: str, generator: str) -> str | None:
    """Returns the MAC aux-head target class for one manifest row, or None
    if this row should contribute no aux loss (all real-labeled rows).

    `label` is "real" or "ai" (manifest.csv's own convention, see
    scripts/fetch_clip_train_set.py). `generator` is that row's `generator`
    column value -- meaningful only when label == "ai"; ignored otherwise.
    """
    if label != "ai":
        return None

    name = (generator or "").lower()
    for kw in _GAN_KEYWORDS:
        if kw in name:
            return GAN
    for kw in _DIFFUSION_KEYWORDS:
        if kw in name:
            return DIFFUSION
    return OTHER_AI


def collapse_rare_classes(aux_labels: list[str | None], floor: int = 20) -> dict[str, str]:
    """Given the aux labels actually produced by bucket_generator_family()
    for one manifest (None entries ignored), returns a remapping
    {rare_class: "other-ai"} for any class with fewer than `floor` samples.

    Applied at training time against the in-memory label list, not baked
    into manifest.csv -- so the floor can be tuned (e.g. after a fetch
    yields more or fewer samples of some family) without re-fetching data.

    Returns an empty dict (no remapping needed) if every class already
    clears the floor. Prints a warning and expects the caller to disable
    the aux branch entirely if the resulting number of distinct classes
    (after collapsing) is fewer than 2 -- that check is the caller's
    responsibility since it also needs to affect model construction
    (n_aux_classes), not just labels.
    """
    counts = Counter(a for a in aux_labels if a is not None)
    remap: dict[str, str] = {}
    for cls, n in counts.items():
        if cls != OTHER_AI and n < floor:
            remap[cls] = OTHER_AI
    return remap


def surviving_aux_classes(aux_labels: list[str | None], floor: int = 20) -> list[str]:
    """Convenience wrapper: applies collapse_rare_classes() and returns the
    sorted list of aux classes that remain after collapsing. Prints a loud
    warning (does not raise) if fewer than 2 classes survive -- callers
    (training scripts) must check len(...) < 2 themselves and disable the
    aux branch for that run rather than train on a degenerate one-class
    target."""
    remap = collapse_rare_classes(aux_labels, floor=floor)
    counts = Counter(a for a in aux_labels if a is not None)
    survivors = sorted({remap.get(cls, cls) for cls in counts})
    if len(survivors) < 2:
        print(
            f"WARNING: only {len(survivors)} generator-family class(es) survive "
            f"collapse_rare_classes(floor={floor}): {survivors}. The MAC aux "
            "branch has no viable multi-class target on this data -- the "
            "caller must disable it for this training run rather than fit a "
            "one-class classifier."
        )
    return survivors
