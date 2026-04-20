"""Incepedia — synthetic pretraining dataset aimed at outperforming Cosmopedia.

Public API lives in submodules:
    incepedia.config      — path constants, env loading
    incepedia.generation  — OpenRouter async batch generator
    incepedia.dedup       — MinHash + embedding dedup
    incepedia.decontam    — three-layer benchmark decontamination
    incepedia.training    — nanotron training launchers
    incepedia.eval        — lighteval task defs
"""

__version__ = "0.0.1"
