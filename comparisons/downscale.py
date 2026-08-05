"""Uniform down-scaling baseline: resize every image to a fixed token budget."""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ratio", type=float, required=True, help="token budget, e.g. 0.32")
    parser.add_argument("--write-images", type=Path)
    args = parser.parse_args()

    # Down-scaling is prepare_labels with the boxes left in the resized frame.
    import subprocess, sys
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "tools" / "prepare_labels.py"),
           "--input", str(args.input), "--output", str(args.output),
           "--image-ratio", str(args.ratio), "--label-ratio", "1.0"]
    if args.write_images:
        cmd += ["--write-images", str(args.write_images)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
