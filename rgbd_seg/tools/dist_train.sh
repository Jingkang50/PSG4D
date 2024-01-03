#!/usr/bin/env bash

CONFIG=configs/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic.py
WORK_DIR=work_dirs/pvsg_demo_mask2former_r50_default_train_dsit
GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-15}
PORT=${PORT:-$((29500 + $RANDOM % 29))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


if command -v torchrun &> /dev/null
then
  echo "Using torchrun mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nnodes=$NNODES \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
else
  echo "Using launch mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --work-dir ${WORK_DIR} --launcher pytorch ${@:3}
fi



