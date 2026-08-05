#!/usr/bin/env bash
# Evaluate one split and score it overall and per object scale.
#   bash scripts/eval.sh context        # also: uncommon, ego4d
set -euo pipefail

SPLIT="${1:-context}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO/configs/qwen2_5vl_3b_lora_predict_egoint_lite.yaml}"

case "$SPLIT" in
    context|uncommon) DATASET="egointention_${SPLIT}_test_10to50" ;;
    ego4d)            DATASET="mllm_paco_ego4d_v1_test_10to50" ;;
    *)                echo "unknown split '$SPLIT'"; exit 1 ;;
esac
OUTPUT="$REPO/outputs/eval_${SPLIT}"

python "$REPO/tools/run_eval.py" "$CONFIG" \
    --eval-dataset "$DATASET" --output-dir "$OUTPUT" \
    --nproc "${NPROC:-4}" --master-port "${MASTER_PORT:-29540}"

python "$REPO/tools/score_per_scale.py" \
    --predictions "$OUTPUT/generated_predictions.jsonl" \
    --dataset "${DATA_DIR:-$REPO/data}/${DATASET}.json"
