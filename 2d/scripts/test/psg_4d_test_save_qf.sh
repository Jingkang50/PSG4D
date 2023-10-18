#!/usr/bin/env bash

set -x

PARTITION=priority
JOB_NAME=test_tune_default_tracking
CONFIG=configs/unitrack/psg_4d_unitrack.py
CHECKPOINT=work_dirs/gta_rgbd_train_only/epoch_8.pth
WORK_DIR=work_dirs/gta_rgbd_test_save_qf_train_only
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}

PYTHONPATH="$(/mnt/lustre/wxpeng/OpenPVSG $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u tools/test_query_tube.py ${CONFIG} ${CHECKPOINT} --work-dir ${WORK_DIR} --launcher="slurm" ${PY_ARGS}
