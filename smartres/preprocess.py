"""Build the low- and high-resolution views of one image."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image


@dataclass
class DualResolutionImage:
    low_res_pixels: torch.Tensor    # [N_lo, C * temporal * patch^2]
    low_res_grid: torch.Tensor      # [1, 3] as (t, h, w) in patches
    high_res_pixels: torch.Tensor
    high_res_grid: torch.Tensor

    @property
    def linear_ratio(self) -> float:
        return float(self.high_res_grid[0, 1]) / float(self.low_res_grid[0, 1])


def build_dual_resolution(
    image: Image.Image,
    processor,
    hr_scale: float = 0.2,
    max_pixels: Optional[int] = None,
) -> DualResolutionImage:
    """Prepare ``image`` at both resolutions. ``hr_scale`` is low-res / high-res."""
    if not 0 < hr_scale <= 1:
        raise ValueError(f"hr_scale must be in (0, 1], got {hr_scale}")

    image = image.convert("RGB")
    factor = processor.patch_size * processor.merge_size

    high_res = processor(images=image, return_tensors="pt")

    # Match the high-resolution geometry the processor settled on, then shrink it.
    limit = max_pixels or getattr(processor, "max_pixels", None) or 12845056
    width, height = image.size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > limit:
        beta = math.sqrt((height * width) / limit)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor

    shrink = math.sqrt(hr_scale)
    low_h = max(factor, math.ceil(h_bar * shrink / factor) * factor)
    low_w = max(factor, math.ceil(w_bar * shrink / factor) * factor)

    low_res = processor(
        images=image.resize((low_w, low_h), Image.BICUBIC), return_tensors="pt"
    )

    return DualResolutionImage(
        low_res_pixels=low_res["pixel_values"],
        low_res_grid=low_res["image_grid_thw"],
        high_res_pixels=high_res["pixel_values"],
        high_res_grid=high_res["image_grid_thw"],
    )


def attach_high_res_view(
    mm_inputs: dict, processor, images=None, hr_scale: float = 0.2
) -> dict:
    """Collator hook: put ``pixel_frames_hr`` / ``hr_grid_thw`` into ``mm_inputs``.

    Uses the views the image processor stashed if it builds both; otherwise builds them
    here from ``images``, so no custom image processor is required.
    """
    import numpy as np

    image_processor = getattr(processor, "image_processor", processor)
    stashed = getattr(image_processor, "_last_processed_hr_frames", None)
    if stashed is not None:
        for stash, key in (
            ("_last_processed_hr_frames", "pixel_frames_hr"),
            ("_last_processed_hr_grid_thw", "hr_grid_thw"),
        ):
            value = getattr(image_processor, stash, None)
            if value is not None:
                mm_inputs[key] = torch.as_tensor(np.ascontiguousarray(np.stack(value, axis=0)))
            setattr(image_processor, stash, None)
        return mm_inputs

    if not images:
        return mm_inputs

    # mm_inputs already holds the low-resolution view; add the high-resolution one.
    pixels, grids = [], []
    for image in images:
        views = build_dual_resolution(image, image_processor, hr_scale=hr_scale)
        pixels.append(views.high_res_pixels)
        grids.append(views.high_res_grid)
    mm_inputs["pixel_frames_hr"] = torch.cat(pixels, dim=0)
    mm_inputs["hr_grid_thw"] = torch.cat(grids, dim=0)
    return mm_inputs
