#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

PRETRAIN_CKPT="./model/LFD_test/checkpoint-last.pth"
IMAGENET_PATH="./training_data/lfd_dataset/"
OUTPUT_DIR="./result/lfd_rgt_test"
CONDITION="fx","hrz"
TARGET="rgt"

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=29505 \
main.py \
--model LFD-B/32 \
--proj_dropout 0.0 \
--P_mean -1.0 --P_std 0.8 \
--img_size 512 --noise_scale 0.2 \
--batch_size 128 --blr 5e-5 \
--epochs 1201 --warmup_epochs 5 \
--output_dir ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} \
--class_num 1 \
--pretrained_base ${PRETRAIN_CKPT} 
# --resume ${OUTPUT_DIR} \