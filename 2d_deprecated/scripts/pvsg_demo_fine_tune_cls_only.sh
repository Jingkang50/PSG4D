set -x

PARTITION=dsta
JOB_NAME=pvsg_finetune_default
CONFIG=configs/mask2former/mask2former_r50_pvsg_image_panoptic_tune_cls_head.py
WORK_DIR=work_dirs/pvsg_demo_mask2former_r50_tune_cls_slurm
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-7}
node=65
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
