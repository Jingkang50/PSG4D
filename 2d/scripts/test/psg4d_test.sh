#!/usr/bin/env bash
# sh scripts/test/psg4d_test.sh

set -x

PARTITION=priority
JOB_NAME=test_acc
CONFIG=configs/mask2former/mask2former_dual_r50_rgbd_test.py
CHECKPOINT=work_dirs/gta_rgbd_train_only/epoch_8.pth
WORK_DIR=work_dirs/gta_rgbd_test
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
PY_ARGS=${@:5}

PYTHONPATH="$(/mnt/lustre/wxpeng/OpenPVSG $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} \
    --work-dir ${WORK_DIR} --eval mAP --launcher="slurm" ${PY_ARGS}
