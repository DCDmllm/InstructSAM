#!/bin/bash

# ⛔️ 1. Cancel conda auto-activation
unset CONDA_SHLVL
unset CONDA_EXE
unset _CE_CONDA
unset CONDA_PREFIX
unset CONDA_PROMPT_MODIFIER
unset CONDA_PYTHON_EXE
unset CONDA_DEFAULT_ENV

# ⛔️ 2. Remove conda from PATH
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'anaconda3' | paste -sd ':' -)

# ✅ 3. Proceed with the actual training logic
echo "Conda has been disabled. Running training script..."

export PYTHONWARNINGS="ignore"
# export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_PATH=${1:-"instructsam_2b"}

ARG_WORLD_SIZE=${2:-1}
ARG_NPROC_PER_NODE=${3:-8} 

ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16669
ARG_RANK=${4:-0}

if [ -z "${WORLD_SIZE:-}" ]; then
  WORLD_SIZE="$ARG_WORLD_SIZE"
fi
if [ -z "${NPROC_PER_NODE:-}" ]; then
  NPROC_PER_NODE="$ARG_NPROC_PER_NODE"
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  MASTER_ADDR="$ARG_MASTER_ADDR"
fi
if [ -z "${MASTER_PORT:-}" ]; then
  MASTER_PORT="$ARG_MASTER_PORT"
fi
if [ -z "${RANK:-}" ]; then
  RANK="$ARG_RANK"
fi


echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MODEL_PATH: $MODEL_PATH"


SAVE_DIR=./evaluation_results
DATA_ROOT=./data

SPLIT_LIST=("val" "test")

for SPLIT in "${SPLIT_LIST[@]}"; do
    echo "run reason_seg_${SPLIT}..."
    DATASET="reason_seg_${SPLIT}"
    QUESTION_FILE="${DATA_ROOT}/eval/${DATASET}.json"

    torchrun --nnodes="$WORLD_SIZE" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --node_rank="$RANK" \
        evaluation/eval_reasonseg.py \
        --model_path "work_dirs/${MODEL_PATH}" \
        --question_file "${QUESTION_FILE}" \
        --image_folder "${DATA_ROOT}" \
        --output_file "${SAVE_DIR}/$MODEL_PATH/${DATASET}.json"
done