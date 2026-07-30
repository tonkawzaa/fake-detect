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
    # IPTC digitalSourceType codes (http://cv.iptc.org/newscodes/digitalsourcetype/*)
    # -- substring match also catches "compositeWithTrainedAlgorithmicMedia".
    "trainedalgorithmicmedia",
)
