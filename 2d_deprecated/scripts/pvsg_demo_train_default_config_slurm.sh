#!/bin/bash
# sh scripts/psgtr/psgtr_train_r50.sh
# check gpu usage:
# srun -p dsta -w SG-IDC1-10-51-2-73 nvidia-smi
# squeue -w SG-IDC1-10-51-2-73

# GPU=4
# CPU=7
# node=73
# PORT=32000
# jobname=pvsg_finetune_default

# PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
# python -m torch.distributed.launch \
# --nproc_per_node=${GPU} \
# --master_port=$PORT \
# tools/train.py \
# configs/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic.py \
# --work-dir work_dirs/pvsg_demo_mask2former_r50_default_train_slurm \
# --gpus ${GPU} \
# --launcher pytorch


set -x

PARTITION=dsta
JOB_NAME=pvsg_finetune_default
CONFIG=configs/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic.py
WORK_DIR=work_dirs/pvsg_demo_mask2former_r50_default_train_slurm
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-7}
node=72
PY_ARGS=${@:5}

PYTHONPATH="$(/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-${node} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
