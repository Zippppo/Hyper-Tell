#!/bin/bash

cd /home/xiaoqingguo/Rongkun/code/Hyper-Tell

MASTER_PORT="${MASTER_PORT:-29602}"

CUDA_VISIBLE_DEVICES=7 torchrun --master-port="${MASTER_PORT}" --nproc_per_node=1 Body-Tell/train.py \
    --config Body-Tell/configs/test-model-config.yaml \
    --epochs 10 \
    --batch-size 8 \
    --lr 1e-4 \
    --num-workers 4 \
    --checkpoint-dir Body-Tell/checkpoints/0526S2I-aligned128 \
    --grad-clip-norm 1.0 \
    --wandb \
    --wandb-offline \
    --wandb-project Body-Tell \
    --wandb-tags phase2