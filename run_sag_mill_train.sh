#!/bin/bash
MASTER_PORT="29500"
MASTER_ADDR="localhost"

RESULT_DIR="./results"
CUDA_VISIBLE_DEVICES="4, 5, 6, 7"

DATASET_NAME="SAGMill"
NTRAINING_STEPS=60000
NVALID_STEPS=1000

DATA_PATH="/mnt/raid0sata1/gnn/Rot_drum_mono"
MODEL_PATH="${RESULT_DIR}/models/${DATASET_NAME}/"
ROLLOUT_PATH="${RESULT_DIR}/rollouts/${DATASET_NAME}/"
logout_path="${RESULT_DIR}/logs/${DATASET_NAME}/"

echo "Training"
# python -m gns.train --data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="latest" --train_state_file="latest" --ntraining_steps=${NTRAINING_STEPS} --nvalid_steps=${NVALID_STEPS} --mode='train' --exp_id=$2
CUDA_VISIBLE_DEVICES="4, 5, 6, 7" torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
./gns/train_baseline.py \
--data_path=${DATA_PATH} --model_path=${MODEL_PATH} --model_file="latest" \
--train_state_file="latest" --ntraining_steps=${NTRAINING_STEPS} \
--nvalid_steps=${NVALID_STEPS} --mode='train' --exp_id=$1 --is_cuda 1 \
--log_path=${logout_path}

# echo "Rollout"
# CUDA_VISIBLE_DEVICES="4, 5, 6, 7" torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
#     --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
#     ./gns/train_baseline.py \
#     --data_path=${DATA_PATH} \
#     --model_path=${MODEL_PATH} --model_file="-model-60000.pt" --output_path=${ROLLOUT_PATH} \
#     --is_cuda 1 \
#     --mode='rollout' \
#     --exp_id=$3