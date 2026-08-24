#!/usr/bin/env bash
# Evaluate one split and score it overall, per object scale, and by token Ratio.
#   bash scripts/eval.sh context        # also: uncommon
set -euo pipefail

SPLIT="${1:-context}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO/configs/qwen2_5vl_3b_lora_predict_egoint_lite.yaml}"

case "$SPLIT" in
    context|uncommon) DATASET="egointention_${SPLIT}_test_10to50" ;;
    *)                echo "unknown split '$SPLIT'"; exit 1 ;;
esac
OUTPUT="$REPO/outputs/eval_${SPLIT}"
mkdir -p "$OUTPUT"

# The vision tower emits one keyed record per forward, which score_token_ratio.py reads.
SMARTRES_TOKEN_LOG=1 python "$REPO/tools/run_eval.py" "$CONFIG" \
    --eval-dataset "$DATASET" --output-dir "$OUTPUT" \
    --nproc "${NPROC:-4}" --master-port "${MASTER_PORT:-29540}" 2>&1 | tee "$OUTPUT/run.log"

# 10to50 stores boxes in the 10% frame while the image is at 50%, so the scale buckets
# need both in one frame; without it every object is classified five times too small.
python "$REPO/tools/score_per_scale.py" \
    --predictions "$OUTPUT/generated_predictions.jsonl" \
    --dataset "${DATA_DIR:-$REPO/data}/${DATASET}.json" \
    --label-ratio 0.2

python "$REPO/tools/score_token_ratio.py" \
    --log "$OUTPUT/run.log" --extract "$OUTPUT/tokens.txt" \
    --full-dataset "${DATA_DIR:-$REPO/data}/egointention_${SPLIT}_test.json" \
    --high-res-dataset "${DATA_DIR:-$REPO/data}/${DATASET}.json"
