#!/usr/bin/env bash
# Run with 0 as the first argument to run on a single node
# Run with localhost as the second argument to indicate only one node is used
set -x  # print the commands

# Uncomment when no longer debugging
export CUDA_LAUNCH_BLOCKING=1

# FOR LARGER RUNS, we should change num_workers to 10?

export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes

# change output directory
OUTPUT_DIR=''  # Your output folder for deepspeed config file, logs and checkpoints

# Create csv for training
# video_path, start frame, stop frame
DATA_PATH=''  # The data list file path.
# pretrain data list file follows the following format
# for the video data line: video_path, 0, -1
# for the rawframe data line: frame_folder_path, start_index, total_frames, 0

# TODO: Change values before final run
N_NODES=${N_NODES:-1}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-1}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training args

# Please refer to `run_mae_pretraining.py` for the meaning of the following hyperreferences
# Eventually add \ --use_wandb  to the script
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=$1 --master_addr=$2 \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 16 \
        --with_checkpoint \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 4 \
        --num_workers 1 \
        --opt adamw \
        --lr 6e-4 \
        --clip_grad 0.02 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 30 \
        --save_ckpt_freq 2 \
        --epochs 40 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}
