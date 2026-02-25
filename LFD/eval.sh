#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT 

DATA_PATH="/home/guozhixiang/data/2dAll/"
CONDITION="fx","hrz"
TARGET="rgt"

CKPT_DIR="./result/lfd_rgt_10abshrzloss_0.1bending/"

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=29506 \
main.py \
--model LFD-B/32 \
--img_size 512 --noise_scale 0.1 \
--gen_bsz 50 --num_images 50 \
--num_sampling_steps 50 \
--cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} \
--data_path ${DATA_PATH} \
--cond_in_ch 2 \
--cond ${CONDITION} \
--target ${TARGET} \
--resume ${CKPT_DIR} \
--class_num 1 \
--evaluate_gen