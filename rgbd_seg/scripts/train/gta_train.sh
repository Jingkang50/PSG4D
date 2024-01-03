set -x

# sh scripts/train/gta_train.sh
PARTITION=priority
JOB_NAME=gta
CONFIG=configs/mask2former/mask2former_dual_rgbd.py
WORK_DIR=work_dirs/gta_rgbd_train
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}

PYTHONPATH="/mnt/lustre/jkyang/PSG4D/code/rgbd_seg":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    -x SG-IDC2-10-51-5-51 \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
