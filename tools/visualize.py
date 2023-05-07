import argparse
import os
import os.path as osp
import time
import warnings
import pickle

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

import sys
sys.path.append("/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image")
from datasets.datasets.builder import build_dataset

if __name__ == "__main__":
    cfg_file = "/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image/configs/mask2former/mask2former_r50_pvsg_image_panoptic_tune_cls_head.py"
    cfg = Config.fromfile(cfg_file)
    

# build model
cfg.model.train_cfg = None
# cfg.model.type = 'Mask2Former'
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
# epoch2 (bs=32)
checkpoint = "/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image/work_dirs/pvsg_demo_mask2former_r50_tune_cls_slurm/epoch_1.pth"
checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

# dataset
dataset = build_dataset(cfg.data.val)
model.CLASSES = dataset.CLASSES
test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
data_loader = build_dataloader(dataset, **test_loader_cfg)

model = build_ddp(
            model,
            'cuda',
            device_ids=[0,1,2,3],
            broadcast_buffers=False)
outputs = multi_gpu_test(model, data_loader)

with open("/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image/cls_fine_tune_ckpt1_results.pickle", "wb") as f:
    pickle.dump(outputs, f)
