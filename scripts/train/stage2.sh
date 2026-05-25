echo "MASTER_ADDR=$MASTER_ADDR"
n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "node rank:" $SLURM_PROCID

if [[ "${SLURM_PROCID}" -eq 0 ]]; then
    echo "==== GPU Model on node rank 0 ===="
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo "==============================================="
fi


echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

WORK_DIR=work_dirs
RUN_NAME=instructsam_stage2_2b
OUTPUT_DIR=$WORK_DIR/$RUN_NAME

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

MODEL_ARGS=(
    --model_path work_dirs/instructsam_stage1_merged
    --mask_decoder_model checkpoints/sam3
    --gradient_checkpointing True
    --use_liger_kernel False
    --loss_sample_points True
)

DATA_ARGS=(
    --ann_path ./data/stage2.txt
    --data_root ./data
    --data_path_root ./data/training/
    --data_cache_dir ./data/cache
    --model_max_length 16384
    --mm_max_length 8192
    --fps 2
    --max_frames 512
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 1
    --num_train_epochs 1
    --remove_unused_columns False
    --use_multi_objs True
    --skip_none False
)

OPTIMIZER_ARGS=(
    --llm_lr 2e-6
    --projector_lr 2e-6
    --vision_encoder_lr 2e-6
    --sam_decoder_lr 5e-6
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
)

TRAINING_ARGS=(
    --deepspeed scripts/zero1.json
    --bf16 True
    --lora_enable True
    --bf16 True
    --tf32 True
    --fp16 False
    --dataloader_num_workers 16
    --loss_reduction_scope batch
    --average_tokens_across_devices False
    --group_by_modality_length True
)

LOG_ARGS=(
    --output_dir $OUTPUT_DIR
    --run_name $RUN_NAME
    --logging_steps 1
    --report_to "wandb"
    --save_strategy "steps"
    --save_steps 1000
    --save_total_limit 2
)

set -x

torchrun --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --nproc_per_node 8 \
    --master_addr $MASTER_ADDR \
    --master_port 25031 \
    -m instructsam.train \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOG_ARGS[@]} 2>&1 | tee -a logs/${RUN_NAME}_${SLURM_PROCID}.log
