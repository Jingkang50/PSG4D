set -x

PARTITION=dsta
JOB_NAME=pvsg_v1_finetune_default
CONFIG=configs/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic.py
WORK_DIR=work_dirs/pvsg_v1_finetune_default_repeat1
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
node=73
PY_ARGS=${@:5}

PYTHONPATH="$(/mnt/lustre/wxpeng/OpenPVSG $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-${node} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
