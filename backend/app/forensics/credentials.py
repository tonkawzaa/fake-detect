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

from .ai_hints import GENERATIVE_AI_HINTS


@dataclass
class C2paReport:
    present: bool
    claim_generator: str | None = None
    is_generative_ai: bool = False
    actions: list[str] = field(default_factory=list)
    raw: dict | None = None


def _manifest_hints(manifest: dict) -> tuple[str | None, list[str], list[str], list[str]]:
    """Pulls claim_generator, action names, digitalSourceType values, and
    softwareAgent values out of one manifest's c2pa.actions assertions.
    softwareAgent is the field the C2PA spec actually uses to name the tool
    that performed an action (e.g. "Dreamina/7.5.0" on a c2pa.created
    action) -- an earlier version of this function never read it at all,
    so a real generative-AI claim naming its own producer there was
    silently invisible to the haystack/is_ai check below."""
    claim_generator = manifest.get("claim_generator") or manifest.get("claim_generator_info", [{}])[0].get("name")
    actions, source_types, software_agents = [], [], []
    for assertion in manifest.get("assertions", []):
        if assertion.get("label", "").startswith("c2pa.actions"):
            for action in assertion.get("data", {}).get("actions", []):
                if action.get("action"):
                    actions.append(action["action"])
                if action.get("digitalSourceType"):
                    source_types.append(action["digitalSourceType"])
                if action.get("softwareAgent"):
                    software_agents.append(action["softwareAgent"])
    return claim_generator, actions, source_types, software_agents


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

    # Walk the full ingredient chain, not just the active manifest. Many
    # real export/share pipelines wrap the original AI-generation claim in
    # a second manifest that only records a generic re-packaging step (seen
    # empirically: a Dreamina-generated PNG's active manifest had
    # claim_generator "c2pa-tool/0.1.0 c2pa-rs/0.39.0" and a single
    # "c2pa.transcoded" action -- nothing AI-related at all -- while the
    # actual "c2pa.created" action naming softwareAgent "Dreamina/7.5.0"
    # lived in a *parent* manifest reachable only via the active manifest's
    # `ingredients[].active_manifest` reference). Checking only the active
    # manifest missed this entirely and reported 0% AI probability despite
    # C2PA credentials clearly being present. `visited` guards against a
    # cyclic ingredient graph looping forever.
    visited: set[str] = set()
    to_visit = [active_label] if active_label else [manifest.get("label")]
    all_claim_generators: list[str] = []
    actions: list[str] = []
    source_types: list[str] = []
    software_agents: list[str] = []

    while to_visit:
        label = to_visit.pop()
        if not label or label in visited or label not in manifests:
            continue
        visited.add(label)
        m = manifests[label]
        cg, acts, srcs, agents = _manifest_hints(m)
        if cg:
            all_claim_generators.append(cg)
        actions.extend(acts)
        source_types.extend(srcs)
        software_agents.extend(agents)
        for ingredient in m.get("ingredients", []):
            parent_label = ingredient.get("active_manifest") or ingredient.get("manifest_data", {}).get("identifier")
            if parent_label:
                to_visit.append(parent_label)

    haystack = " ".join(filter(None, [*all_claim_generators, *actions, *source_types, *software_agents])).lower()
    is_ai = any(hint in haystack for hint in GENERATIVE_AI_HINTS)

    # Prefer surfacing whichever claim_generator/softwareAgent actually
    # matched a generative-AI hint over the active manifest's own
    # claim_generator -- the latter is frequently just the generic tool
    # that re-packaged/exported the image (see above), not the tool that
    # produced it, so showing it alone would leave the UI's "Generative AI
    # claim: Yes" next to a "Claim generator" field that names the wrong
    # (or an uninformative) tool.
    display_generator = manifest.get("claim_generator") or manifest.get("claim_generator_info", [{}])[0].get("name")
    for candidate in [*software_agents, *all_claim_generators]:
        if candidate and any(hint in candidate.lower() for hint in GENERATIVE_AI_HINTS):
            display_generator = candidate
            break

    return C2paReport(
        present=True,
        claim_generator=display_generator,
        is_generative_ai=is_ai,
        actions=actions,
        raw=data,
    )
