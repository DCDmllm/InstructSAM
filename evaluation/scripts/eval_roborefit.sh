#!/bin/bash
export PYTHONWARNINGS="ignore"

# RoboRefIt Benchmark Evaluation Script
# Usage: bash evaluation/scripts/eval_roborefit.sh [MODEL_PATH] [WORLD_SIZE] [NPROC_PER_NODE] [RANK]

MODEL_PATH=${1:-"instructsam_2b"}

ARG_WORLD_SIZE=${2:-1}
ARG_NPROC_PER_NODE=${3:-4}

ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16669
ARG_RANK=${4:-0}

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "========================================="
echo "RoboRefIt Benchmark Evaluation"
echo "========================================="
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MODEL_PATH: $MODEL_PATH"
echo "========================================="
echo ""

SAVE_DIR=./evaluation_results
DATA_ROOT=./data

# RoboRefIt has two test sets: testA and testB
DATASET_LIST=("roborefit_testA" "roborefit_testB")

for DATASET in "${DATASET_LIST[@]}"; do
    echo "========================================="
    echo "Evaluating ${DATASET}..."
    echo "========================================="
    
    QUESTION_FILE="${DATA_ROOT}/eval/${DATASET}_qa.json"
    OUTPUT_FILE="${SAVE_DIR}/${MODEL_PATH}/${DATASET}.json"
    
    # Check if question file exists
    if [ ! -f "$QUESTION_FILE" ]; then
        echo "❌ Error: Question file not found: $QUESTION_FILE"
        echo "Please check the data path."
        continue
    fi
    
    # Run evaluation
    torchrun --nnodes="$WORLD_SIZE" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --node_rank="$RANK" \
        evaluation/eval_roborefit.py \
        --model_path "./work_dirs/${MODEL_PATH}" \
        --question_file "${QUESTION_FILE}" \
        --image_folder "${DATA_ROOT}" \
        --output_file "${OUTPUT_FILE}" \
        --batch-size 1 \
        --num-workers 8
    
    if [ $? -eq 0 ]; then
        echo "✅ ${DATASET} evaluation completed"
        echo "Results saved to: ${OUTPUT_FILE}"
    else
        echo "❌ ${DATASET} evaluation failed"
    fi
    echo ""
done

