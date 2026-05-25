#!/bin/bash
export PYTHONWARNINGS="ignore"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_PATH=${1:-"0117_ft_v8_lora_bs64"}

ARG_WORLD_SIZE=${3:-1}
ARG_NPROC_PER_NODE=${4:-2}

ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16669
ARG_RANK=${5:-0}

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi


echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MODEL_PATH: $MODEL_PATH"


SAVE_DIR=./evaluation_results
DATA_ROOT=./data
DATASET="inst2seg"
# QUESTION_FILE="${DATA_ROOT}/eval/ours/ego_merge.json"
QUESTION_FILE="${DATA_ROOT}/eval/${DATASET}.json"


torchrun --nnodes="$WORLD_SIZE" \
    --nproc_per_node=8 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank="$RANK" \
    evaluation/eval_inst2seg.py \
    --model_path "work_dirs/${MODEL_PATH}" \
    --question_file "${QUESTION_FILE}" \
    --image_folder "${DATA_ROOT}" \
    --output_file "${SAVE_DIR}/$MODEL_PATH/${DATASET}.json"