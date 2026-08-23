"""Install SmartRes onto a constructed Qwen2.5-VL model."""

import os
import sys
from types import MethodType
from typing import Optional

import torch

from ..forward import smartres_vision_forward
from ..router import Router

TOKEN_RECORD_KEY = "[smartres-tokens]"


def _emit_token_record(output) -> None:
    """One keyed line per forward, for tools/score_token_ratio.py to pull out of the log."""
    print(
        f"{TOKEN_RECORD_KEY} samples={len(output.lengths)} assembled={sum(output.lengths)} "
        f"encoded={output.high_res_encoded} hr_total={output.high_res_total} "
        f"activated={output.activated_ratio:.6f}",
        file=sys.stderr, flush=True,
    )


def install_smartres(
    model,
    tau: float = 0.5,
    router_layer: int = 30,
    encode_snap: str = "window",
    margin: float = 1.0,
    lambda_route: float = 1.0,
    pos_weight: float = 5.0,
    lambda_hinge: float = 5.0,
    router_state_dict: Optional[dict] = None,
):
    """Attach the router and swap the vision forward. Returns ``model``."""
    visual = getattr(model, "visual", None)
    if visual is None:
        raise AttributeError(
            "model has no .visual -- install_smartres expects a Qwen2.5-VL model"
        )
    if not 0 < router_layer <= len(visual.blocks):
        raise ValueError(
            f"router_layer must be in (0, {len(visual.blocks)}], got {router_layer}"
        )

    visual.router = Router(
        embed_dim=visual.config.hidden_size,
        tau=tau,
        margin=margin,
        lambda_route=lambda_route,
        pos_weight=pos_weight,
        lambda_hinge=lambda_hinge,
    ).to(device=next(visual.parameters()).device, dtype=next(visual.parameters()).dtype)

    if router_state_dict is not None:
        missing, unexpected = visual.router.load_state_dict(router_state_dict, strict=False)
        if missing:
            raise ValueError(f"router weights missing: {missing}")
        if unexpected:
            raise ValueError(
                f"unexpected keys in router weights: {unexpected}. Convert a legacy "
                f"checkpoint with tools/convert_checkpoint.py first."
            )

    def forward(self, pixel_values, grid_thw, **kwargs):
        """Signature-compatible with the stock vision forward."""
        high_res_pixels = kwargs.get("pixel_frames_hr")
        high_res_grid = kwargs.get("hr_grid_thw")
        if high_res_pixels is None or high_res_grid is None:
            raise ValueError(
                "SmartRes needs the high-resolution frame (pixel_frames_hr / hr_grid_thw). "
                "The image processor must be configured to emit both views."
            )
        # Routing supervision comes from the target boxes, which arrive as text.
        route_target = kwargs.get("route_target")
        if route_target is None and self.training and kwargs.get("text_prompt"):
            from ..target import build_routing_target, parse_boxes
            boxes = [parse_boxes(str(t)) for t in kwargs["text_prompt"]]
            route_target = build_routing_target(boxes, grid_thw, self.config.patch_size)

        output = smartres_vision_forward(
            self,
            low_res_pixels=pixel_values,
            low_res_grid=grid_thw,
            high_res_pixels=high_res_pixels,
            high_res_grid=high_res_grid,
            router=self.router,
            router_layer=router_layer,
            encode_snap=encode_snap,
            route_target=route_target,
        )
        # Kept on the module so the training loop can add the routing terms to the loss and
        # the benchmark can read the activation rate, without widening the return.
        self.last_vision_output = output
        if os.environ.get("SMARTRES_TOKEN_LOG"):
            _emit_token_record(output)
        # For training loops that read a single auxiliary loss off the vision tower.
        if output.loss_route is not None:
            self.loss_mts = (self.router.lambda_route * output.loss_route
                             + self.router.lambda_hinge * output.loss_hinge)
        return output.tokens, output.lengths

    visual.forward = MethodType(forward, visual)
    visual.smartres_config = {
        "tau": tau,
        "router_layer": router_layer,
        "encode_snap": encode_snap,
    }
    return model


@torch.no_grad()
def routing_losses(model):
    """The routing terms from the most recent forward, or ``(None, None)``."""
    output = getattr(getattr(model, "visual", None), "last_vision_output", None)
    if output is None:
        return None, None
    return output.loss_route, output.loss_hinge
