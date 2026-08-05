#!/usr/bin/env python
"""Rescale a grounding dataset to an image resolution and a label coordinate frame."""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Tuple

BOX = re.compile(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]")


def resize_for_ratio(
    width: int, height: int, token_ratio: float,
    patch_size: int = 14, merge_size: int = 2, max_pixels: int = 12845056,
) -> Tuple[int, int, float, float]:
    """Target size for a token budget, and the box scale factors that go with it."""
    factor = patch_size * merge_size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor

    shrink = math.sqrt(token_ratio)
    new_h = max(factor, math.ceil(h_bar * shrink / factor) * factor)
    new_w = max(factor, math.ceil(w_bar * shrink / factor) * factor)
    return new_w, new_h, new_w / width, new_h / height


def rescale_boxes(text: str, scale_w: float, scale_h: float) -> str:
    def replace(match: "re.Match") -> str:
        x1, y1, x2, y2 = (float(v) for v in match.groups())
        return f"[{x1 * scale_w:.2f}, {y1 * scale_h:.2f}, {x2 * scale_w:.2f}, {y2 * scale_h:.2f}]"

    return BOX.sub(replace, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-ratio", type=float, required=True,
                        help="image token budget relative to the original (1.0 = unchanged)")
    parser.add_argument("--label-ratio", type=float, default=1.0,
                        help="label frame relative to the RESIZED image (1.0 = same frame)")
    parser.add_argument("--image-root", type=Path,
                        help="rewrite image paths to this directory, keeping the filename")
    parser.add_argument("--write-images", type=Path,
                        help="also write the resized images here (needs pillow)")
    args = parser.parse_args()

    for name, value in (("image-ratio", args.image_ratio), ("label-ratio", args.label_ratio)):
        if not 0 < value <= 1:
            raise SystemExit(f"--{name} must be in (0, 1], got {value}")

    from PIL import Image

    with open(args.input) as handle:
        dataset = json.load(handle)

    if args.write_images:
        args.write_images.mkdir(parents=True, exist_ok=True)

    converted, skipped = [], 0
    for sample in dataset:
        images = sample.get("images") or []
        if not images:
            skipped += 1
            continue
        try:
            with Image.open(images[0]) as image:
                width, height = image.size
                image_w, image_h, image_sw, image_sh = resize_for_ratio(width, height, args.image_ratio)
                # The label frame is relative to the already-resized image.
                _, _, label_sw, label_sh = resize_for_ratio(image_w, image_h, args.label_ratio)
                if args.write_images:
                    destination = args.write_images / Path(images[0]).name
                    image.convert("RGB").resize((image_w, image_h), Image.BICUBIC).save(destination, quality=95)
        except Exception:
            skipped += 1
            continue

        scale_w, scale_h = image_sw * label_sw, image_sh * label_sh
        sample = json.loads(json.dumps(sample))          # don't mutate the input
        for message in sample.get("messages", []):
            if message.get("role") == "assistant":
                message["content"] = rescale_boxes(message["content"], scale_w, scale_h)

        if args.write_images:
            sample["images"] = [str(args.write_images / Path(images[0]).name)]
        elif args.image_root:
            sample["images"] = [str(args.image_root / Path(images[0]).name)]
        converted.append(sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(converted, handle, ensure_ascii=False)

    print(f"in      : {len(dataset)} samples")
    print(f"out     : {len(converted)} -> {args.output}")
    if skipped:
        print(f"skipped : {skipped} (missing or unreadable image)")
    print(f"image   : {args.image_ratio:.0%} of the original token budget")
    print(f"labels  : {args.label_ratio:.0%} frame relative to that image "
          f"({args.image_ratio * args.label_ratio:.1%} of the original)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
