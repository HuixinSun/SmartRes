"""Runtime patches for Qwen2.5-VL: install the router, splice the variable-length span."""

from .integration import install_runtime, spliced_rope_index
from .patch import install_smartres
from .processor import DualResolutionImageProcessor
from .sequence import SplicedBatch, realign_labels, splice_visual_sequence

__all__ = [
    "install_smartres",
    "install_runtime",
    "DualResolutionImageProcessor",
    "spliced_rope_index",
    "splice_visual_sequence",
    "realign_labels",
    "SplicedBatch",
]
