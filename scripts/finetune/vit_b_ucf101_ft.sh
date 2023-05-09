#!/usr/bin/env bash
# Run with 0 as the first argument to run on a single node
# Run with localhost as the second argument to indicate only one node is used
set -x  # print the commands

export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes

ckpt_epoch=$3 # Checkpoint to load from pre-training and to use for fine-tuning (how many pre-training epochs)

OUTPUT_DIR="./work_dir/vit_b_ucf101_ft_${ckpt_epoch}e_ucf101_ft"  # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH='/home/datasets/ucf101/video_mae_splits/fine-tune/' # Fine-tune on val set to test pre-training on train set
# finetune data list file follows the following format
# for the video data line: video_path, label
# for the rawframe data line: frame_folder_path, total_frames, label
MODEL_PATH="./work_dir/vit_b_ucf101_pt_40e/checkpoint-${ckpt_epoch}.pth"  # Model for initializing parameters

N_NODES=${N_NODES:-1}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-1}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:4}  # Other training args

# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=$1 --master_addr=$2 \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set UCF101 \
        --nb_classes 101 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 5 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 1 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 5 \
        --epochs 20 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}