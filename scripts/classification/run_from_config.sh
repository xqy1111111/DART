#!/usr/bin/env bash
set -euo pipefail

# Run classification token-pruning experiments from a JSON config
# Usage: ./scripts/classification/run_from_config.sh experiments/classification_config.json [DATA_ROOT]

CONFIG_PATH=${1:-experiments/classification_config.json}
DATA_ROOT=${2:-data}

RESULTS_ROOT=$(jq -r '.save.results_root' "$CONFIG_PATH")
CKPT_ROOT=$(jq -r '.save.checkpoints_root' "$CONFIG_PATH")
IMG_SIZE=$(jq -r '.eval.img_size' "$CONFIG_PATH")
BATCH_SIZE=$(jq -r '.eval.batch_size' "$CONFIG_PATH")
NUM_WORKERS=$(jq -r '.eval.num_workers' "$CONFIG_PATH")
SEED=$(jq -r '.eval.seed' "$CONFIG_PATH")
MODEL=$(jq -r '.model' "$CONFIG_PATH")
METHODS=($(jq -r '.methods[]' "$CONFIG_PATH"))
# Prefer reduction_ratios; fallback to ratios for backward compatibility
if jq -e '.reduction_ratios' "$CONFIG_PATH" >/dev/null; then
  RATIOS=($(jq -r '.reduction_ratios[]' "$CONFIG_PATH"))
else
  RATIOS=($(jq -r '.ratios[]' "$CONFIG_PATH"))
fi
PIVOT_IMAGE=$(jq -r '.pivot_image_token // 8' "$CONFIG_PATH")
PIVOT_TEXT=$(jq -r '.pivot_text_token // 0' "$CONFIG_PATH")
MAX_TRUNC=$(jq -r '.max_num_trunction // 0' "$CONFIG_PATH")
TRAIN_EPOCHS=$(jq -r '.train.epochs' "$CONFIG_PATH")
FREEZE_BACKBONE=$(jq -r '.train.freeze_backbone' "$CONFIG_PATH")
LR=$(jq -r '.train.lr' "$CONFIG_PATH")
WD=$(jq -r '.train.weight_decay' "$CONFIG_PATH")
PRUNE_POLICY_TYPE=$(jq -r '.prune_layer_policy.type' "$CONFIG_PATH")
PRUNE_POLICY_VALUE=$(jq -r '.prune_layer_policy.value' "$CONFIG_PATH")
GPU_LIST=($(jq -r '.compute.gpu_list[]' "$CONFIG_PATH"))

mkdir -p "$RESULTS_ROOT" "$CKPT_ROOT" "$DATA_ROOT"

# Download datasets with robust curl retry and tar validation
LEN=$(jq '.datasets | length' "$CONFIG_PATH")
for ((i=0; i<LEN; i++)); do
  NAME=$(jq -r ".datasets[$i].name" "$CONFIG_PATH")
  URL=$(jq -r ".datasets[$i].download.url" "$CONFIG_PATH")
  if [ ! -d "$DATA_ROOT/$NAME" ]; then
    echo "Downloading $NAME ..."
    pushd "$DATA_ROOT" >/dev/null
    FNAME=$(basename "$URL")
    # Retry downloads and show progress
    curl -L --retry 10 --retry-all-errors --connect-timeout 30 -o "$FNAME" "$URL"
    # Validate archive
    if ! tar tf "$FNAME" >/dev/null 2>&1; then
      echo "Corrupted archive $FNAME, retrying with aria2c if available..."
      if command -v aria2c >/dev/null 2>&1; then
        aria2c -x 16 -s 16 -k 1M "$URL" -o "$FNAME"
        tar tf "$FNAME" >/dev/null 2>&1 || { echo "Archive validation failed for $FNAME"; exit 1; }
      else
        echo "aria2c not found and archive invalid. Please install aria2c or check network."; exit 1
      fi
    fi
    tar xf "$FNAME"
    popd >/dev/null
  fi
  echo "Dataset ready: $DATA_ROOT/$NAME"

done

# Derive prune layer index by policy
# For ViT-Base 12 blocks, relative 0.5 -> layer 6
if [ "$PRUNE_POLICY_TYPE" = "relative" ]; then
  if [[ "$MODEL" == *"vit_base"* ]]; then
    PRUNE_LAYER=$(python - <<'PY'
import math
print(int(round(12*0.5)))
PY
)
  else
    PRUNE_LAYER=6
  fi
else
  PRUNE_LAYER=$PRUNE_POLICY_VALUE
fi

# One-time warmup to cache timm weights (avoid parallel download stalls)
CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python - <<'PY' 2>/dev/null || true
import timm
m = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
print('timm weights cached OK')
PY

# Export training envs
export TRAIN_EPOCHS=$TRAIN_EPOCHS

GPU_COUNT=${#GPU_LIST[@]}
JOB_ID=0
PIDS=()

for ((i=0; i<LEN; i++)); do
  NAME=$(jq -r ".datasets[$i].name" "$CONFIG_PATH")
  RES_DIR="$RESULTS_ROOT/$NAME"
  CK_DIR="$CKPT_ROOT/$NAME"
  mkdir -p "$RES_DIR" "$CK_DIR"

  # baseline
  gpu=${GPU_LIST[$((JOB_ID % GPU_COUNT))]}
  echo "[GPU $gpu] $NAME baseline $MODEL"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/classification/imagenette_tokenprune.py \
    --data-dir "$DATA_ROOT/$NAME" \
    --results-dir "$RES_DIR" \
    --checkpoint-dir "$CK_DIR" \
    --model $MODEL --method none --reduction-ratio 0.0 \
    --pivot-image-token $PIVOT_IMAGE --pivot-text-token $PIVOT_TEXT --max-num-trunction $MAX_TRUNC \
    --prune-layer $PRUNE_LAYER --img-size $IMG_SIZE --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --seed $SEED &
  PIDS+=($!)
  JOB_ID=$((JOB_ID+1))

  # methods x ratios
  for method in "${METHODS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      gpu=${GPU_LIST[$((JOB_ID % GPU_COUNT))]}
      echo "[GPU $gpu] $NAME $MODEL $method ratio=$ratio"
      CUDA_VISIBLE_DEVICES=$gpu python scripts/classification/imagenette_tokenprune.py \
        --data-dir "$DATA_ROOT/$NAME" \
        --results-dir "$RES_DIR" \
        --checkpoint-dir "$CK_DIR" \
        --model $MODEL --method $method --reduction-ratio $ratio \
        --pivot-image-token $PIVOT_IMAGE --pivot-text-token $PIVOT_TEXT --max-num-trunction $MAX_TRUNC \
        --prune-layer $PRUNE_LAYER --img-size $IMG_SIZE --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --seed $SEED &
      PIDS+=($!)
      JOB_ID=$((JOB_ID+1))
    done
  done

done

for pid in "${PIDS[@]}"; do
  wait $pid
done

echo "All experiments completed. Results in $RESULTS_ROOT, checkpoints in $CKPT_ROOT"

