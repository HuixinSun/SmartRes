#!/usr/bin/env python
"""Run a SmartRes config through the patched LLaMA-Factory driver."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

def translate(config: dict) -> tuple:
    """The driver accepts the SmartRes names directly; only batch size is enforced here."""
    if config.get("per_device_eval_batch_size", 1) != 1:
        raise SystemExit(
            "per_device_eval_batch_size must be 1: SmartRes emits a routing-dependent "
            "number of visual tokens, and batching over the resulting left-padded "
            "sequences corrupts decoded boxes without raising an error."
        )
    return dict(config), {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC", 4)))
    parser.add_argument("--master-port", default=os.environ.get("MASTER_PORT", "29540"))
    parser.add_argument("--max-samples", type=int, help="truncate the split, for smoke tests")
    parser.add_argument("--adapter", help="override adapter_name_or_path")
    parser.add_argument("--eval-dataset", help="override eval_dataset")
    parser.add_argument("--dataset-dir", help="override dataset_dir")
    parser.add_argument("--output-dir", help="override output_dir")
    parser.add_argument("--dataset", help="override dataset (training)")
    parser.add_argument("--tau", type=float, help="routing threshold, M = STE(S > tau)")
    parser.add_argument("--router-layer", type=int, help="vision block the router reads")
    parser.add_argument("--encode-snap", choices=["window", "unit"], help="encode-set granularity")
    parser.add_argument("--lr-budget", type=float, help="r_LR, low-resolution token budget")
    parser.add_argument("--hr-budget", type=float, help="r_HR, high-resolution token budget")
    parser.add_argument("--lambda-hinge", type=float, help="weight of the margin regulariser")
    parser.add_argument("--epochs", type=float, help="num_train_epochs")
    parser.add_argument("--dry-run", action="store_true", help="print the translated config and stop")
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    for flag, key in (("tau", "tau"), ("router_layer", "router_layer"),
                      ("encode_snap", "encode_snap"), ("lr_budget", "lr_budget"),
                      ("hr_budget", "hr_budget"), ("lambda_hinge", "lambda_hinge"),
                      ("dataset", "dataset"), ("epochs", "num_train_epochs")):
        value = getattr(args, flag, None)
        if value is not None:
            config[key] = value

    driver, env = translate(config)
    if args.max_samples:
        driver["max_samples"] = args.max_samples
    if args.adapter:
        driver["adapter_name_or_path"] = args.adapter
    if args.dataset_dir:
        driver["dataset_dir"] = args.dataset_dir
    if args.eval_dataset:
        driver["eval_dataset"] = args.eval_dataset
    if args.output_dir:
        driver["output_dir"] = args.output_dir

    # Batched generation over left-padded variable-length visual spans corrupts decoded
    # boxes without raising, so refuse rather than produce quiet garbage.
    if driver.get("per_device_eval_batch_size", 1) != 1:
        raise SystemExit("per_device_eval_batch_size must be 1")

    print("[smartres] translated config:")
    for key in sorted(driver):
        print(f"    {key}: {driver[key]}")
    print("[smartres] environment:")
    for key in sorted(env):
        print(f"    {key}={env[key]}")
    if args.dry_run:
        return 0

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(driver, handle, sort_keys=True)
        translated = handle.name

    try:
        return subprocess.call(
            [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node", str(args.nproc),
                "--master_port", str(args.master_port),
                "-m", "llamafactory.launcher", translated,
            ],
            env={**os.environ, **env},
        )
    finally:
        os.unlink(translated)


if __name__ == "__main__":
    raise SystemExit(main())
