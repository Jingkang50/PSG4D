set -x

PARTITION=priority
JOB_NAME=psg4d_train_gta_rgbd
CONFIG=configs/mask2former/mask2former_dual_r50_rgbd.py
WORK_DIR=work_dirs/gta_rgbd_train_only
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}

PYTHONPATH="$(/mnt/lustre/wxpeng/OpenPVSG $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
