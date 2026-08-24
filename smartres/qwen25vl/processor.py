"""An image processor that emits both resolutions SmartRes needs from one image."""

import math
from typing import Optional

import numpy as np
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor


class DualResolutionImageProcessor(Qwen2VLImageProcessor):
    """Returns the low-resolution view, and stashes the high-resolution one for the collator.

    ``pixel_values`` / ``image_grid_thw`` describe the view the router reads; the frame the
    router selects from is left on ``_last_processed_hr_frames`` / ``_last_processed_hr_grid_thw``.
    """

    def __init__(self, *args, hr_scale: float = 0.2, **kwargs):
        # `high_res_scale` is the name the training configs use for the same quantity.
        hr_scale = kwargs.pop("high_res_scale", hr_scale)
        for unused in ("use_multi_scale", "scale_levels", "conf_thresh", "scale_thresh",
                       "base_resolution"):
            kwargs.pop(unused, None)
        super().__init__(*args, **kwargs)
        if not 0 < hr_scale <= 1:
            raise ValueError(f"hr_scale must be in (0, 1], got {hr_scale}")
        self.hr_scale = hr_scale
        self._last_processed_hr_frames = None
        self._last_processed_hr_grid_thw = None

    def _low_res_size(self, width: int, height: int, factor: int) -> tuple:
        """The low-resolution pixel size, in the frame the tower will actually see."""
        limit = self.max_pixels or 12845056
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > limit:
            beta = math.sqrt((height * width) / limit)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        shrink = math.sqrt(self.hr_scale)
        return (
            max(factor, math.ceil(w_bar * shrink / factor) * factor),
            max(factor, math.ceil(h_bar * shrink / factor) * factor),
        )

    def preprocess(self, images, videos=None, **kwargs):
        from transformers.image_utils import make_list_of_images

        if videos is not None:
            return super().preprocess(images=images, videos=videos, **kwargs)

        images = make_list_of_images(images)
        factor = self.patch_size * self.merge_size
        low_batches, hr_frames, hr_grids = [], [], []

        for image in images:
            image = image.convert("RGB") if hasattr(image, "convert") else image
            high = super().preprocess(images=image, **kwargs)
            hr_frames.append(np.asarray(high["pixel_values"]))
            hr_grids.append(np.asarray(high["image_grid_thw"])[0])

            width, height = image.size
            low_w, low_h = self._low_res_size(width, height, factor)
            resample = getattr(self, "resample", None)
            shrunk = image.resize((low_w, low_h), resample) if resample else image.resize((low_w, low_h))
            low_batches.append(super().preprocess(images=shrunk, do_resize=False, **kwargs))

        # The collator reads these off the processor and puts them in the model inputs.
        self._last_processed_hr_frames = hr_frames
        self._last_processed_hr_grid_thw = hr_grids

        merged = low_batches[0]
        if len(low_batches) > 1:
            merged.data["pixel_values"] = np.concatenate(
                [np.asarray(b["pixel_values"]) for b in low_batches], axis=0
            )
            merged.data["image_grid_thw"] = np.concatenate(
                [np.asarray(b["image_grid_thw"]) for b in low_batches], axis=0
            )
        return merged
