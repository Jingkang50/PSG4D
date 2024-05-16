#!/usr/bin/env bash

CONFIG=configs/mask2former/mask2former_dual_rgbd.py
WORK_DIR=work_dirs/gta_rgbd_train
GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29505}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nnodes=$NNODES \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --nproc_per_node=$GPUS \
    tools/train.py ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --launcher pytorch
