srun -p dsta --mpi=pmi2 --gres=gpu:2 -n1 --ntasks-per-node=1 --job-name=wx_ood --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-73 \
python tools/train.py \
configs/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic.py \
--work-dir work_dirs/pvsg_demo_mask2former_r50_default_train_2gpu \
--launcher slurm

