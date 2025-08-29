
DATASET=$2
EPOCH=$4
LR=$6
PER_DEV_BATCH_SIZE=$8
MODEL_PATH=${10}
NUM_NODES=${12}
TRAIN_ON_PROMPT=${14}
SAVE_STEP=${16}
WEIGHT_ALPHA=${18}
PORT=${20}

MON=$(date +%m)     # Current month
DAY=$(date +%d)     # Current day
HOUR=$(date +%H)    # Current hour
MIN=$(date +%M)     # Current minute

# Generate identifier with minute included
IDENTIFIER="${MON}_${DAY}_${HOUR}_${MIN}_${DATASET}_lr_${LR}_epoch_${EPOCH}"


WANDB_PROJECT=DecoupleSearch torchrun --nproc_per_node $NUM_NODES --master_port $PORT  src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path  $MODEL_PATH\
    --dataset $DATASET \
    --template chatml \
    --finetuning_type full \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --output_dir save/$IDENTIFIER \
    --overwrite_cache \
    --cutoff_len 6200 \
    --preprocessing_num_workers 32 \
    --weight_alpha $WEIGHT_ALPHA \
    --per_device_train_batch_size $PER_DEV_BATCH_SIZE \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --overwrite_output_dir \
    --logging_steps 5 \
    --save_steps $SAVE_STEP \
    --learning_rate $LR \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.005 \
    --num_train_epochs $EPOCH \
    --plot_loss \
    --train_on_prompt $TRAIN_ON_PROMPT \
    --run_name $IDENTIFIER\
    --report_to wandb \
    --save_only_model \
    --bf16