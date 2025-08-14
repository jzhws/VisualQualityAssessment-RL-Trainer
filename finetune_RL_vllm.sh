#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH
export WANDB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH=".:$PYTHONPATH"
OUTDIR=/fs-computility/ResearchEval/shared/zihengjia/GRPO/Scoring_direct_8_12_image_0.5_standard

export DEBUG_MODE="true"
export LOG_PATH="./logs/${WANDB_NAME}.log"
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=7
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))


#using cuda:7 for vllm generation
DISTRIBUTED_ARGS="
    --nproc_per_node 7 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port 25808 \
"


torchrun \
    --nproc_per_node 7 \
    --nnodes 1 \
    --node_rank 0 \
    --master_port 25808 \
    src/open_r1/grpo.py \
    --deepspeed training_scripts/zero3.json \
    --output_dir $OUTDIR \
    --model_name_or_path   "Qwen2.5-VL-7B-Instruct" \
    --train_data_path /tos-bjml-researcheval/jiaziheng/RL4VQA/VideoChat-R1-main_mix_image_vllm/RL_scoring_image.json \
    --eval_data_path /tos-bjml-researcheval/jiaziheng/RL4VQA/VideoChat-R1-main_mix/RL_scoring.json \
    --video_folder /tos-bjml-researcheval/jiaziheng/VQA++/LSVQ \
    --dataset_name xxx \
    --max_prompt_length 8192 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation None \
    --num_train_epochs 5 \
    --run_name $WANDB_NAME \
    --report_to tensorboard \
    --save_steps 100 \
    --save_total_limit 1 \
    --use_vllm true \
    --save_only_model false