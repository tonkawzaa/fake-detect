"""
Phase 4: C2PA content credentials.

Many generators (and increasingly some phone cameras) now sign output with
C2PA manifests. A valid manifest naming a generative-AI producer is
near-conclusive; its absence proves nothing (most images have none yet).

Deliberately out of scope: ELA (unreliable, invites false confidence) and
PRNU (needs many images from the same sensor -- useless on one upload).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

GENERATIVE_AI_HINTS = (
    "midjourney", "dall-e", "dalle", "stable diffusion", "firefly", "openai",
    "generative", "diffusion", "gpt", "imagen", "flux",
)


@dataclass
class C2paReport:
    present: bool
    claim_generator: str | None = None
    is_generative_ai: bool = False
    actions: list[str] = field(default_factory=list)
    raw: dict | None = None


def read_c2pa(image_bytes: bytes, suffix: str = ".jpg") -> C2paReport:
    import c2pa

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        try:
            with c2pa.Reader(tmp.name) as reader:
                manifest_json = reader.json()
        except Exception:
            # No JUMBF/manifest data, unsupported format, or a malformed
            # manifest -- all treated the same as "no credentials found".
            return C2paReport(present=False)

    try:
        data = json.loads(manifest_json)
    except json.JSONDecodeError:
        return C2paReport(present=False)

    manifests = data.get("manifests", {})
    active_label = data.get("active_manifest")
    manifest = manifests.get(active_label) if active_label else next(iter(manifests.values()), None)
    if not manifest:
        return C2paReport(present=False, raw=data)

    claim_generator = manifest.get("claim_generator") or manifest.get("claim_generator_info", [{}])[0].get("name")
    actions = []
    for assertion in manifest.get("assertions", []):
        if assertion.get("label", "").startswith("c2pa.actions"):
            for action in assertion.get("data", {}).get("actions", []):
                if action.get("action"):
                    actions.append(action["action"])

    haystack = " ".join(filter(None, [claim_generator, *actions])).lower()
    is_ai = any(hint in haystack for hint in GENERATIVE_AI_HINTS)

    return C2paReport(
        present=True,
        claim_generator=claim_generator,
        is_generative_ai=is_ai,
        actions=actions,
        raw=data,
    )
