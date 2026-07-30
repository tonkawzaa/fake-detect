"""
Substring hints for detecting a generative-AI provenance claim, shared
between C2PA manifest parsing (credentials.py) and raw XMP/IPTC metadata
parsing (metadata.py) -- the same generator names and IPTC digitalSourceType
codes can appear in either place independently of the other.
"""

from __future__ import annotations

GENERATIVE_AI_HINTS = (
    "midjourney", "dall-e", "dalle", "stable diffusion", "firefly", "openai",
    "generative", "diffusion", "gpt", "imagen", "flux",
    # dreamina (ByteDance/CapCut's image generator, softwareAgent "Dreamina/x.y.z"
    # in C2PA action assertions) -- added after a real Dreamina-generated PNG's
    # C2PA credentials were present but silently unmatched, see
    # credentials.py's read_c2pa() docstring for the (separate) ingredient-chain
    # bug that was also required to actually see this string.
    "dreamina",
    # IPTC digitalSourceType codes (http://cv.iptc.org/newscodes/digitalsourcetype/*)
    # -- substring match also catches "compositeWithTrainedAlgorithmicMedia".
    "trainedalgorithmicmedia",
)
