#!/usr/bin/env bash
set -euo pipefail

# One-shot pipeline: download two datasets (ImageNette, ImageWoof) and run 20 experiments total
# - Each dataset runs 10 groups: baseline (no pruning) + 3 methods (dart/random/knorm) x 3 ratios (0.667/0.778/0.889)
# - Uses up to 8 GPUs in parallel
# Usage: ./scripts/classification/run_two_datasets_20.sh [DATA_ROOT]
# Default DATA_ROOT=./data

DATA_ROOT=${1:-data}
RESULTS_ROOT=${RESULTS_ROOT:-results}
CKPT_ROOT=${CKPT_ROOT:-checkpoints}
IMG_SIZE=${IMG_SIZE:-224}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
PRUNE_LAYER=${PRUNE_LAYER:-6}
SEED=${SEED:-42}

# Prepare directories
mkdir -p "$DATA_ROOT" "$RESULTS_ROOT" "$CKPT_ROOT"

# Download ImageNette (160px) and ImageWoof (160px) if missing
pushd "$DATA_ROOT" >/dev/null
if [ ! -d imagenette2-160 ]; then
  echo "Downloading ImageNette 160px ..."
  wget -q https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz && tar xf imagenette2-160.tgz
fi
if [ ! -d imagewoof2-160 ]; then
  echo "Downloading ImageWoof 160px ..."
  wget -q https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz && tar xf imagewoof2-160.tgz
fi
popd >/dev/null

echo "Datasets ready under: $DATA_ROOT/{imagenette2-160,imagewoof2-160}"

# Experiment grids
DATASETS=(imagenette2-160 imagewoof2-160)
MODEL=vit_base_patch16_224
METHODS=(dart random knorm)
RATIOS=(0.667 0.778 0.889)

GPU_LIST=(0 1 2 3 4 5 6 7)
GPU_COUNT=${#GPU_LIST[@]}
JOB_ID=0
PIDS=()

run_job() {
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES=$gpu python scripts/classification/imagenette_tokenprune.py "$@"
}

START_TS=$(date +%s)

for ds in "${DATASETS[@]}"; do
  DATA_DIR="$DATA_ROOT/$ds"
  RES_DIR="$RESULTS_ROOT/$ds"
  CKPT_DIR="$CKPT_ROOT/$ds"
  mkdir -p "$RES_DIR" "$CKPT_DIR"

  # Baseline (no pruning)
  gpu=${GPU_LIST[$((JOB_ID % GPU_COUNT))]}
  echo "[GPU $gpu] $ds baseline $MODEL"
  run_job $gpu \
    --data-dir "$DATA_DIR" \
    --results-dir "$RES_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --model $MODEL --method none --ratio 0.0 \
    --prune-layer $PRUNE_LAYER --img-size $IMG_SIZE --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --seed $SEED &
  PIDS+=($!)
  JOB_ID=$((JOB_ID+1))

  # 3 methods x 3 ratios
  for method in "${METHODS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      gpu=${GPU_LIST[$((JOB_ID % GPU_COUNT))]}
      echo "[GPU $gpu] $ds $MODEL $method ratio=$ratio"
      run_job $gpu \
        --data-dir "$DATA_DIR" \
        --results-dir "$RES_DIR" \
        --checkpoint-dir "$CKPT_DIR" \
        --model $MODEL --method $method --ratio $ratio \
        --prune-layer $PRUNE_LAYER --img-size $IMG_SIZE --batch-size $BATCH_SIZE --num-workers $NUM_WORKERS --seed $SEED &
      PIDS+=($!)
      JOB_ID=$((JOB_ID+1))
    done
  done

done

# Wait for all jobs to finish
for pid in "${PIDS[@]}"; do
  wait $pid
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "All 20 experiments finished in ${ELAPSED}s"

echo "Results CSVs:"
for ds in "${DATASETS[@]}"; do
  echo "  $RESULTS_ROOT/$ds/imagenette_results.csv"
  ls -lh "$RESULTS_ROOT/$ds/imagenette_results.csv" || true
  echo "  Checkpoints dir: $CKPT_ROOT/$ds"
  ls -1 "$CKPT_ROOT/$ds" | wc -l | xargs echo "  (#files)"
  echo
done

