#!/usr/bin/env python
"""Vision-encoder latency and token ratio, overall and per object scale."""

import argparse
import json
import time
from typing import List, Optional

import torch
from PIL import Image

BUCKETS = (("small", 0.0, 0.005), ("medium", 0.005, 0.05), ("large", 0.05, float("inf")))


def parse_box(text: str) -> Optional[List[float]]:
    import re

    match = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', text)
    if not match:
        match = re.search(r"\[([\d.\s,]+)\]", text)
    if not match:
        return None
    parts = [p for p in match.group(1).replace(",", " ").split() if p]
    return [float(v) for v in parts[:4]] if len(parts) >= 4 else None


def ground_truth(sample: dict) -> Optional[str]:
    for message in sample.get("messages", []):
        if message.get("role") == "assistant":
            return message.get("content")
    return None


def bucket_of(box: List[float], width: int, height: int) -> Optional[str]:
    area = abs(box[2] - box[0]) * abs(box[3] - box[1])
    scale = area / float(width * height)
    for name, low, high in BUCKETS:
        if low <= scale < high:
            return name
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--per-bucket", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--router-layer", type=int, default=30)
    parser.add_argument("--encode-snap", default="window", choices=["window", "unit"])
    parser.add_argument("--hr-scale", type=float, default=0.2)
    parser.add_argument("--out", help="write the measurements as json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[error] a GPU is required for meaningful timings")
        return 2

    from peft import PeftModel
    from transformers import AutoImageProcessor, Qwen2_5_VLForConditionalGeneration

    from smartres import install_smartres
    from smartres.preprocess import build_dual_resolution

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    install_smartres(
        model, tau=args.tau, router_layer=args.router_layer, encode_snap=args.encode_snap
    )
    model.eval()
    visual = model.visual

    with open(args.dataset) as handle:
        dataset = json.load(handle)

    # Pick the first --per-bucket samples of each scale.
    selected = {name: [] for name, _, _ in BUCKETS}
    for sample in dataset:
        if all(len(v) >= args.per_bucket for v in selected.values()):
            break
        images = sample.get("images") or []
        truth = ground_truth(sample)
        if not images or not truth:
            continue
        box = parse_box(truth)
        if box is None:
            continue
        try:
            with Image.open(images[0]) as image:
                name = bucket_of(box, image.width, image.height)
        except Exception:
            continue
        if name and len(selected[name]) < args.per_bucket:
            selected[name].append(images[0])

    def encode_once(views, tau: float) -> float:
        """One vision forward at the given tau; returns milliseconds."""
        visual.router.tau = tau
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            visual(
                views.low_res_pixels.to("cuda", torch.bfloat16),
                views.low_res_grid.to("cuda"),
                pixel_frames_hr=views.high_res_pixels.to("cuda", torch.bfloat16),
                hr_grid_thw=views.high_res_grid.to("cuda"),
            )
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    # tau this low routes every patch, so the encode set is the whole frame.
    FULL_ENCODE_TAU = -1e9

    print(f"warming up ({args.warmup} iterations)...")
    first = next(paths[0] for paths in selected.values() if paths)
    with Image.open(first) as image:
        warm = build_dual_resolution(image, processor, hr_scale=args.hr_scale)
    for _ in range(args.warmup):
        encode_once(warm, args.tau)
        encode_once(warm, FULL_ENCODE_TAU)

    results = {}
    for name, paths in selected.items():
        if not paths:
            continue
        routed_ms, full_ms, activated, encoded = [], [], [], []
        for path in paths:
            with Image.open(path) as image:
                views = build_dual_resolution(image, processor, hr_scale=args.hr_scale)
            routed_ms.append(encode_once(views, args.tau))
            output = visual.last_vision_output
            activated.append(output.activated_ratio)
            encoded.append(output.encode_fraction)
            full_ms.append(encode_once(views, FULL_ENCODE_TAU))

        mean = lambda xs: sum(xs) / len(xs)
        results[name] = {
            "n": len(paths),
            "activated_ratio": mean(activated),
            "encode_fraction": mean(encoded),
            "smartres_ms": mean(routed_ms),
            "full_ms": mean(full_ms),
            "speedup": mean(full_ms) / mean(routed_ms),
        }

    print(f"\nvision encoder, {args.encode_snap} snapping, tau={args.tau}")
    print("  {:<10}{:>6}{:>12}{:>12}{:>12}{:>12}{:>10}".format(
        "scale", "n", "activated", "encoded", "SmartRes", "full", "speedup"))
    print("  " + "-" * 74)
    for name, _, _ in BUCKETS:
        row = results.get(name)
        if not row:
            continue
        print("  {:<10}{:>6}{:>11.1%}{:>12.1%}{:>10.1f}ms{:>10.1f}ms{:>9.2f}x".format(
            name, row["n"], row["activated_ratio"], row["encode_fraction"],
            row["smartres_ms"], row["full_ms"], row["speedup"]))

    if results:
        total = sum(r["n"] for r in results.values())
        weighted = lambda key: sum(r[key] * r["n"] for r in results.values()) / total
        print("  " + "-" * 74)
        print("  {:<10}{:>6}{:>11.1%}{:>12.1%}{:>10.1f}ms{:>10.1f}ms{:>9.2f}x".format(
            "overall", total, weighted("activated_ratio"), weighted("encode_fraction"),
            weighted("smartres_ms"), weighted("full_ms"),
            weighted("full_ms") / weighted("smartres_ms")))

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"config": vars(args), "results": results, "timestamp": time.time()},
                      handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
