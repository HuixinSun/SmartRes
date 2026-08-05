#!/usr/bin/env bash
# Train the router and the LoRA adapter together. Any flag below overrides the config.
#
#   bash scripts/train.sh
#   bash scripts/train.sh --tau 0.4 --hr-budget 1.00 --epochs 1
#
#   --tau            routing threshold, M = STE(S > tau). Lower routes more patches.
#   --router-layer   vision block the router reads (30 of 32)
#   --encode-snap    window | unit -- how far the encode set is rounded up
#   --lr-budget      r_LR, low-resolution token budget   (Lite/Pro: 0.10)
#   --hr-budget      r_HR, high-resolution token budget  (Lite 0.50, Pro 1.00)
#   --lambda-hinge   weight of the margin regulariser
#   --dataset        training set name under data/
#   --epochs         number of epochs
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO/configs/qwen2_5vl_3b_lora_sft_egoint_lite.yaml}"

python "$REPO/tools/run_eval.py" "$CONFIG" \
    --nproc "${NPROC:-2}" --master-port "${MASTER_PORT:-29530}" \
    ${OUTPUT:+--output-dir "$OUTPUT"} "$@"
