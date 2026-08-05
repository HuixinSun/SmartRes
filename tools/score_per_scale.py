#!/usr/bin/env python
"""Score grounding predictions overall and per object scale (P_s / P_m / P_l)."""

import argparse
import json
import re
import sys
from typing import List, Optional, Tuple

BUCKETS = (("small", 0.0, 0.005), ("medium", 0.005, 0.05), ("large", 0.05, float("inf")))


def strip_fences(text: str) -> str:
    text = text.strip()
    start = text.find("```json")
    if start == -1 and "```" in text:
        start = text.find("```")
    if start != -1:
        text = text[start:]
    if text.startswith("```"):
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def parse_box(text: str) -> Optional[List[float]]:
    """First ``bbox_2d`` in the response, tolerating the formats the model actually emits."""
    try:
        obj = json.loads(strip_fences(text))
        if isinstance(obj, list) and obj:
            if isinstance(obj[0], dict) and "bbox_2d" in obj[0]:
                box = obj[0]["bbox_2d"]
                if isinstance(box, list) and len(box) >= 4:
                    return [float(v) for v in box[:4]]
            if isinstance(obj[0], list) and len(obj[0]) >= 4:
                return [float(v) for v in obj[0][:4]]
            if len(obj) >= 4 and all(isinstance(v, (int, float)) for v in obj[:4]):
                return [float(v) for v in obj[:4]]
    except Exception:
        pass

    number = r"[\s]*([\d.]+)[\s]*"
    four = re.search(rf"\[{number},{number},{number},{number}\]", text)
    if four:
        return [float(v) for v in four.groups()]
    two_points = re.search(rf"\[{number},{number}\].*?\[{number},{number}\]", text)
    if two_points:
        return [float(v) for v in two_points.groups()]
    return None


def iou(a: List[float], b: List[float]) -> float:
    ax1, ax2 = sorted((a[0], a[2]))
    ay1, ay2 = sorted((a[1], a[3]))
    bx1, bx2 = sorted((b[0], b[2]))
    by1, by2 = sorted((b[1], b[3]))
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def ground_truth_turn(sample: dict) -> Optional[str]:
    for message in sample.get("messages", []):
        if message.get("role") == "assistant":
            return message.get("content")
    return None


def image_area(path: str) -> Optional[float]:
    from PIL import Image

    try:
        with Image.open(path) as image:
            return float(image.width * image.height)
    except Exception:
        return None


class Tally:
    __slots__ = ("total", "parsed", "hit_05", "hit_03", "iou_sum")

    def __init__(self) -> None:
        self.total = self.parsed = self.hit_05 = self.hit_03 = 0
        self.iou_sum = 0.0

    def add(self, overlap: Optional[float]) -> None:
        self.total += 1
        if overlap is None:
            return
        self.parsed += 1
        self.iou_sum += overlap
        self.hit_05 += overlap >= 0.5
        self.hit_03 += overlap >= 0.3

    def row(self, valid_only: bool = False) -> Tuple[float, float, float]:
        denominator = (self.parsed if valid_only else self.total) or 1
        return (
            100.0 * self.hit_05 / denominator,
            100.0 * self.hit_03 / denominator,
            self.iou_sum / denominator,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="generated_predictions.jsonl")
    parser.add_argument("--dataset", help="the dataset json, needed for per-scale buckets")
    parser.add_argument("--label-ratio", type=float, default=1.0,
                        help="token budget of the frame the boxes are stored in, relative to "
                             "the image (0.2 for the 10to50 sets)")
    parser.add_argument("--valid-only", action="store_true", help="also print parsed-rows-only figures")
    args = parser.parse_args()

    with open(args.predictions) as handle:
        predictions = [json.loads(line) for line in handle if line.strip()]

    overall = Tally()
    buckets = {name: Tally() for name, _, _ in BUCKETS}
    unscaled = 0

    dataset = None
    if args.dataset:
        with open(args.dataset) as handle:
            dataset = json.load(handle)
        if len(dataset) != len(predictions):
            print(
                f"[error] {len(predictions)} predictions but {len(dataset)} dataset rows. "
                f"Per-scale scoring joins by index and cannot align these.",
                file=sys.stderr,
            )
            return 2
        mismatches = sum(
            (predictions[i].get("label") or "").strip() != (ground_truth_turn(dataset[i]) or "").strip()
            for i in range(min(200, len(dataset)))
        )
        if mismatches:
            print(
                f"[error] labels disagree with the dataset on {mismatches}/200 sampled rows, "
                f"so the files are not in the same order. Re-run scoring against the dataset "
                f"the predictions were produced from.",
                file=sys.stderr,
            )
            return 2

    for index, record in enumerate(predictions):
        truth = parse_box(record.get("label", ""))
        guess = parse_box(record.get("predict", ""))
        overlap = None if (guess is None or truth is None) else iou(guess, truth)
        overall.add(overlap)

        if dataset is None or truth is None:
            continue
        images = dataset[index].get("images") or []
        area = image_area(images[0]) if images else None
        if not area:
            unscaled += 1
            continue
        box_area = abs(truth[2] - truth[0]) * abs(truth[3] - truth[1])
        # Boxes may live in a smaller frame than the stored image; put both in one frame
        # before taking the ratio, or every object looks smaller than it is.
        scale = box_area / (area * args.label_ratio)
        for name, low, high in BUCKETS:
            if low <= scale < high:
                buckets[name].add(overlap)
                break

    def emit(title: str, tally: Tally, valid_only: bool = False) -> None:
        p05, p03, miou = tally.row(valid_only)
        print(f"  {title:<24}{tally.total:>8}{p05:>10.2f}{p03:>10.2f}{miou:>10.4f}")

    print(f"\npredictions : {args.predictions}")
    print(f"parsed      : {overall.parsed}/{overall.total} "
          f"({100.0 * (overall.total - overall.parsed) / max(overall.total, 1):.2f}% unparsable, scored as IoU 0)")
    print("\n  {:<24}{:>8}{:>10}{:>10}{:>10}".format("split", "n", "P@0.5", "P@0.3", "mIoU"))
    print("  " + "-" * 62)
    emit("overall", overall)

    if dataset is not None:
        print()
        for name, low, high in BUCKETS:
            span = f"S < {high}" if low == 0 else (f"S >= {low}" if high == float("inf") else f"{low} <= S < {high}")
            emit(f"{name} ({span})", buckets[name])
        if unscaled:
            print(f"\n  [warn] {unscaled} rows had no readable image and were left out of the buckets")
    else:
        print("\n  (pass --dataset to break this down by object scale)")

    if args.valid_only:
        print("\n  parsed rows only -- for decoder comparisons, not the headline metric")
        emit("overall", overall, valid_only=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
