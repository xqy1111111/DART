#!/bin/bash

CKPT=/mnt/petrelfs/wenzichen/hf_models/llava-v1.5-7b 
MODEL=llava-v1.5-7b

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file /mnt/petrelfs/wenzichen/SparseVLMs/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /mnt/petrelfs/wenzichen/SparseVLMs/playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file /mnt/petrelfs/wenzichen/SparseVLMs/playground/data/eval/MME/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --sparse \
    --attn_implementation flash_attention_2 \
    --pruned_layer 2 \
    --image_token_start_index 35 \
    --image_token_length 576 \
    --reduction_ratio 0.778 \
    --pivot_image_token 4 \
    --pivot_text_token 4 


cd /mnt/petrelfs/wenzichen/SparseVLMs/playground/data/eval/MME

python convert_answer_to_mme.py --experiment $MODEL

cd eval_tool

python calculation.py --results_dir answers/$MODEL
