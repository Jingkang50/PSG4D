#!/usr/bin/env bash

set -x

PARTITION=dsta
JOB_NAME=test_tune_default_tracking
CONFIG=configs/unitrack/imagenet_resnet50_s3_womotion_timecycle.py
CHECKPOINT=work_dirs/pvsg_v1_finetune_default_repeat1/epoch_5.pth
WORK_DIR=work_dirs/pvsg_v1_test_ckpt5_save_qf1_newest
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}
node=73

PYTHONPATH="$(/mnt/lustre/wxpeng/OpenPVSG $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-${node} \
    python -u tools/test_query_tube.py ${CONFIG} ${CHECKPOINT} --work-dir ${WORK_DIR} --launcher="slurm" ${PY_ARGS}
