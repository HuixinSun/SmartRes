"""SmartRes: dynamic resolution routing for efficient egocentric grounding."""

from .assembly import AssembledSequence, order_preserving_assembly
from .forward import VisionOutput, smartres_vision_forward
from .router import Router, RoutingOutput, straight_through_threshold
from .target import build_routing_target, parse_boxes, routing_loss
from .qwen25vl import install_smartres

__version__ = "0.1.0"

__all__ = [
    "install_smartres",
    "Router",
    "RoutingOutput",
    "straight_through_threshold",
    "smartres_vision_forward",
    "VisionOutput",
    "order_preserving_assembly",
    "AssembledSequence",
    "build_routing_target",
    "parse_boxes",
    "routing_loss",
]
