# sh scripts/train_sailvos3d.sh
PARTITION=priority
JOB_NAME=gta
CONFIG=config/DKNet_run1_sailvos3d_100e_dis_train.yaml
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

PYTHONPATH="/mnt/lustre/jkyang/PSG4D/code/pc_seg":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -m torch.distributed.launch train_sailvos3d.py \
    --config ./config/DKNet_run1_sailvos3d_100e_dis_train.yaml

