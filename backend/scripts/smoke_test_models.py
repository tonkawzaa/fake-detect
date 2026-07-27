"""
Phase 0 gate: load every AI-detector candidate, run it on a small known-real /
known-AI face set, and ASSERT label polarity empirically rather than trusting
model cards or notebook comments (Community Forensics' own notebook text
disagreed with what its own example outputs show — see plan notes).

A model that does not beat chance on this set is reported as DROPPED, not
silently averaged into the ensemble later.

Usage:
    uv run python scripts/smoke_test_models.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from app.detectors.registry import build_candidates

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "smoke"
REAL_DIR = DATA_DIR / "real"
AI_DIR = DATA_DIR / "ai"


def load_images(d: Path) -> list[Image.Image]:
    paths = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
    return [Image.open(p).convert("RGB") for p in paths]


def main() -> None:
    real_imgs = load_images(REAL_DIR)
    ai_imgs = load_images(AI_DIR)
    print(f"Smoke set: {len(real_imgs)} known-real, {len(ai_imgs)} known-AI faces\n")
    if not real_imgs or not ai_imgs:
        print("ERROR: smoke set is empty. See data/smoke/{real,ai}/.")
        sys.exit(1)

    results = []
    for adapter in build_candidates():
        print(f"=== {adapter.name} ({adapter.repo_id if hasattr(adapter, 'repo_id') else ''}) ===")
        try:
            t0 = time.time()
            adapter.load()
            load_s = time.time() - t0
        except Exception as e:
            print(f"  LOAD FAILED: {e}\n")
            results.append({"name": adapter.name, "status": "load_failed", "error": str(e)})
            continue

        if hasattr(adapter, "id2label"):
            print(f"  id2label: {adapter.id2label}  (fake_idx={adapter.fake_idx})")

        try:
            t0 = time.time()
            real_scores = [adapter.predict(img) for img in real_imgs]
            ai_scores = [adapter.predict(img) for img in ai_imgs]
            infer_s = (time.time() - t0) / (len(real_imgs) + len(ai_imgs))
        except Exception as e:
            print(f"  PREDICT FAILED: {e}\n")
            results.append({"name": adapter.name, "status": "predict_failed", "error": str(e)})
            continue

        mean_real = sum(real_scores) / len(real_scores)
        mean_ai = sum(ai_scores) / len(ai_scores)
        # simple accuracy at threshold 0.5
        correct = sum(s < 0.5 for s in real_scores) + sum(s >= 0.5 for s in ai_scores)
        acc = correct / (len(real_scores) + len(ai_scores))

        polarity_ok = mean_ai > mean_real
        beats_chance = acc > 0.55  # small margin above coin-flip on a 50-image set

        status = "OK" if (polarity_ok and beats_chance) else "DROP"
        print(f"  mean p_ai on real set : {mean_real:.3f}")
        print(f"  mean p_ai on AI set   : {mean_ai:.3f}")
        print(f"  polarity assertion (mean_ai > mean_real): {'PASS' if polarity_ok else 'FAIL'}")
        print(f"  accuracy @0.5         : {acc:.3f}")
        print(f"  load time             : {load_s:.1f}s   mean inference: {infer_s*1000:.0f}ms/img")
        print(f"  -> {status}\n")

        results.append(
            {
                "name": adapter.name,
                "status": status,
                "mean_real": mean_real,
                "mean_ai": mean_ai,
                "polarity_ok": polarity_ok,
                "accuracy": acc,
            }
        )

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    kept = [r for r in results if r.get("status") == "OK"]
    for r in results:
        print(f"  {r['name']:40s} {r.get('status')}")
    print(f"\n{len(kept)}/{len(results)} candidates kept for the ensemble.")
    if not kept:
        print("WARNING: no model beat chance. Phase 2 needs re-scoping before proceeding.")


if __name__ == "__main__":
    main()
