"""Routing supervision: which low-resolution patches overlap the ground-truth box."""

import json
import re
from typing import List, Optional, Sequence

import torch

BOX = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")


def parse_boxes(text: str) -> List[List[float]]:
    """Every ``[x1, y1, x2, y2]`` in a target string, in low-resolution coordinates."""
    try:
        obj = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())
        if isinstance(obj, list):
            boxes = [o["bbox_2d"] for o in obj if isinstance(o, dict) and "bbox_2d" in o]
            if boxes:
                return [[float(v) for v in b[:4]] for b in boxes]
    except Exception:
        pass
    return [[float(v) for v in m.groups()] for m in BOX.finditer(text)]


def build_routing_target(
    boxes: Sequence[Optional[Sequence[float]]],
    grid_thw: torch.Tensor,
    patch_size: int = 14,
    device=None,
) -> torch.Tensor:
    """[N_lo] in {0, 1}: 1 where a low-resolution patch overlaps any box.

    Boxes are per frame and already in low-resolution pixel coordinates, which is what the
    ``*_10to50`` annotations store.
    """
    device = device or grid_thw.device
    per_frame = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    target = torch.zeros(int(sum(per_frame)), dtype=torch.float32, device=device)

    start = 0
    for index, row in enumerate(grid_thw.tolist()):
        t, height, width = int(row[0]), int(row[1]), int(row[2])
        frame_boxes = boxes[index] if index < len(boxes) else None
        if frame_boxes is None:
            start += per_frame[index]
            continue
        if frame_boxes and not isinstance(frame_boxes[0], (list, tuple)):
            frame_boxes = [frame_boxes]

        rows = torch.arange(height, device=device).view(height, 1).expand(height, width)
        cols = torch.arange(width, device=device).view(1, width).expand(height, width)
        y0, y1 = rows * patch_size, (rows + 1) * patch_size
        x0, x1 = cols * patch_size, (cols + 1) * patch_size

        mask = torch.zeros(height, width, dtype=torch.bool, device=device)
        for box in frame_boxes:
            bx1, by1, bx2, by2 = (float(v) for v in box[:4])
            overlap_w = (torch.clamp(x1, max=bx2) - torch.clamp(x0, min=bx1)).clamp(min=0)
            overlap_h = (torch.clamp(y1, max=by2) - torch.clamp(y0, min=by1)).clamp(min=0)
            mask |= (overlap_w * overlap_h) > 0

        flat = mask.flatten().float()
        for frame in range(t):
            offset = start + frame * height * width
            target[offset:offset + height * width] = flat
        start += per_frame[index]

    return target


def routing_loss(model, lambda_route: float = 1.0, lambda_hinge: float = 5.0):
    """The weighted routing terms from the most recent forward, or None."""
    output = getattr(getattr(model, "visual", None), "last_vision_output", None)
    if output is None or output.loss_route is None:
        return None
    return lambda_route * output.loss_route + lambda_hinge * output.loss_hinge
