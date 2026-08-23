#!/usr/bin/env python
"""Report the visual-token Ratio: SmartRes tokens as a fraction of full-resolution tokens.

Run an evaluation with SMARTRES_TOKEN_LOG=1 so the vision tower emits one keyed record per
forward, then point this at the log:

    SMARTRES_TOKEN_LOG=1 bash scripts/eval.sh context
    python tools/score_token_ratio.py --log outputs/eval_context/run.log \\
        --extract outputs/eval_context/tokens.txt \\
        --full-dataset data/egointention_context_test.json \\
        --high-res-dataset data/egointention_context_test_10to50.json
"""

import argparse
import json
import re
import sys
from typing import Dict, List, Optional

from PIL import Image

KEY = "[smartres-tokens]"
FIELD = re.compile(r"(\w+)=([-\d.]+)")
MERGE_FACTOR = 28  # patch 14 x spatial merge 2


def read_records(path: str) -> List[Dict[str, float]]:
    """Every keyed line in the file, whether it is a raw eval log or an extracted list."""
    records = []
    with open(path, errors="replace") as handle:
        for line in handle:
            start = line.find(KEY)
            if start == -1:
                continue
            fields = {k: float(v) for k, v in FIELD.findall(line[start + len(KEY):])}
            if {"samples", "assembled", "encoded", "hr_total"} <= fields.keys():
                records.append(fields)
    return records


def visual_tokens(path: str) -> Optional[int]:
    """Tokens Qwen2.5-VL would spend on this image at full resolution."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        return None
    resized_h, resized_w = smart_resize(
        height, width, factor=MERGE_FACTOR,
        min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28,
    )
    return (resized_w // MERGE_FACTOR) * (resized_h // MERGE_FACTOR)


def dataset_images(path: str) -> List[str]:
    with open(path) as handle:
        rows = json.load(handle)
    return [row["images"][0] for row in rows if row.get("images")]


def sum_tokens(images: List[str], label: str) -> tuple:
    total = missing = 0
    for index, image in enumerate(images):
        tokens = visual_tokens(image)
        if tokens is None:
            missing += 1
        else:
            total += tokens
        if (index + 1) % 2000 == 0:
            print(f"  ...{label} {index + 1}/{len(images)}", file=sys.stderr, flush=True)
    return total, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", help="evaluation log containing the keyed records")
    source.add_argument("--records", help="a file of already-extracted records")
    parser.add_argument("--extract", help="write the records pulled out of --log here")
    parser.add_argument("--full-dataset", required=True,
                        help="the full-resolution dataset json, which sets the denominator")
    parser.add_argument("--high-res-dataset",
                        help="the dataset actually evaluated; enables the coverage check")
    args = parser.parse_args()

    records = read_records(args.log or args.records)
    if not records:
        print(f"[error] no {KEY} records found. Re-run the evaluation with "
              f"SMARTRES_TOKEN_LOG=1.", file=sys.stderr)
        return 2

    if args.extract:
        with open(args.extract, "w") as handle:
            for record in records:
                handle.write(
                    f"{KEY} samples={int(record['samples'])} "
                    f"assembled={int(record['assembled'])} encoded={int(record['encoded'])} "
                    f"hr_total={int(record['hr_total'])} "
                    f"activated={record.get('activated', float('nan')):.6f}\n"
                )
        print(f"wrote {len(records)} records to {args.extract}")

    samples = int(sum(r["samples"] for r in records))
    assembled = int(sum(r["assembled"] for r in records))
    encoded = int(sum(r["encoded"] for r in records))
    high_res = int(sum(r["hr_total"] for r in records))

    full_images = dataset_images(args.full_dataset)
    # Under DDP the records arrive in an arbitrary order, so this is a ratio of totals and
    # never pairs record i to row i. That makes the coverage check below the only guard.
    if samples != len(full_images):
        print(f"[error] {samples} samples recorded but {len(full_images)} rows in "
              f"{args.full_dataset}. The log does not cover this dataset exactly.",
              file=sys.stderr)
        return 2

    if args.high_res_dataset:
        if any(r["samples"] != 1 for r in records):
            print("  [warn] batched records, skipping the per-image coverage check")
        else:
            evaluated = sorted(int(r["hr_total"]) for r in records)
            # hr_total counts patches, visual_tokens() counts merged tokens: four to one.
            expected = sorted(
                tokens * 4
                for tokens in map(visual_tokens, dataset_images(args.high_res_dataset))
                if tokens
            )
            if evaluated != expected:
                print(f"[error] the recorded high-resolution patch counts are not the ones "
                      f"in {args.high_res_dataset}. The log is from a different run.",
                      file=sys.stderr)
                return 2
            print("  coverage    : records match the evaluated dataset exactly")

    full_tokens, missing = sum_tokens(full_images, "full-res")
    if missing:
        print(f"[error] {missing} images in {args.full_dataset} could not be read.",
              file=sys.stderr)
        return 2

    smartres_tokens = assembled // 4
    print(f"\nrecords     : {len(records)} ({samples} samples)")
    print(f"\n  {'quantity':<38}{'value':>14}")
    print("  " + "-" * 52)
    print(f"  {'SmartRes visual tokens':<38}{smartres_tokens:>14,}")
    print(f"  {'full-resolution visual tokens':<38}{full_tokens:>14,}")
    print(f"  {'Ratio':<38}{100.0 * smartres_tokens / max(full_tokens, 1):>13.2f}%")
    print(f"  {'high-res patches re-encoded':<38}{100.0 * encoded / max(high_res, 1):>13.2f}%")
    print(f"  {'low-res patches routed to high res':<38}"
          f"{100.0 * sum(r.get('activated', 0.0) for r in records) / len(records):>13.2f}%")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
