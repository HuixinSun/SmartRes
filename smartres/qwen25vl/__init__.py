"""Runtime patches for Qwen2.5-VL: install the router, splice the variable-length span."""

from .patch import install_smartres
from .sequence import SplicedBatch, realign_labels, splice_visual_sequence

__all__ = [
    "install_smartres",
    "splice_visual_sequence",
    "realign_labels",
    "SplicedBatch",
]
