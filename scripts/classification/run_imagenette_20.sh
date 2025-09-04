#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/classification/run_imagenette_20.sh /abs/path/to/data/imagenette2-160 [results_dir] [checkpoints_dir]
DATA_ROOT=${1:-data/imagenette2-160}
RESULTS_DIR=${2:-results}
CKPT_DIR=${3:-checkpoints}

# Models, methods, ratios
MODELS=(vit_base_patch16_224 vit_small_patch16_224)
METHODS=(dart random knorm)
RATIOS=(0.667 0.778 0.889)
PRUNE_LAYER=6
SEED=42
BATCH_SIZE=128
IMG_SIZE=224
NUM_WORKERS=8

# Baselines (no pruning) per model: 2 groups

mkdir -p "$RESULTS_DIR" "$CKPT_DIR"

# Record start time
START_TS=$(date +%s)

GPU_LIST=(0 1 2 3 4 5 6 7)
GPU_COUNT=${#GPU_LIST[@]}
JOB_ID=0
PIDS=()

run_job() {
  local gpu=$1
  shift
  CUDA_VISIBLE_DEVICES=$gpu python scripts/classification/imagenette_tokenprune.py \
    --data-dir "$DATA_ROOT" \
    --results-dir "$RESULTS_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    "$@"
}

# Baselines
for model in "${MODELS[@]}"; do
  gpu=${GPU_LIST[$((JOB_ID % GPU_COUNT))]}
  echo "[GPU $gpu] Baseline $model"
  run_job $gpu --model $model --method none --ratio 0.0 --prune-layer $PRUNE_LAYER --img-size $IMG_SIZE --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --seed $SEED &
  PIDS+=($!)
  JOB_ID=$((JOB_ID+1))

done

# 18 jobs = 2 models * 3 methods * 3 ratios
for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      gpu=${GPU_LIST[$((JOB_ID % GPU_COUNT))]}
      echo "[GPU $gpu] $model $method ratio=$ratio"
      run_job $gpu --model $model --method $method --ratio $ratio --prune-layer $PRUNE_LAYER --img-size $IMG_SIZE --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --seed $SEED &
      PIDS+=($!)
      JOB_ID=$((JOB_ID+1))
    done
  done

done

# Wait for all jobs
for pid in "${PIDS[@]}"; do
  wait $pid
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "All experiments finished in ${ELAPSED}s"
