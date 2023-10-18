import os
from pathlib import Path

import copy

import numpy as np

from mmdet.core import INSTANCE_OFFSET
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

from datasets.datasets.utils import SeqObj, PVSGAnnotation, vpq_eval

# a demo version for pvsg dataset
# THING_CLASSES = ['lighter', 'stand', 'flower', 'glass', 'book', 'spoon', 'glasses', 
#                  'microphone', 'scissor', 'towel', 'basket', 'adult', 'child', 'baby', 
#                  'horse', 'cat', 'dog', 'bed', 'sink', 'faucet', 'sofa', 'table', 'chair', 
#                  'light', 'knife', 'fork', 'candle', 'plate', 'bowl', 'bottle', 'cup', 
#                  'rock', 'road', 'grass', 'hat', 'vegetable', 'fruit', 'cake', 'bread', 
#                  'paper', 'bag', 'box', 'cellphone', 'camera', 'ballon', 'toy', 'racket', 
#                  'bat', 'ball', 'fountain', 'bench', 'bike', 'car']
# STUFF_CLASSES = ['fence', 'tree', 'ground', 'pavement', 'floor', 'ceiling', 'wall', 'water']
# BACKGROUND_CLASSES = ['background'] # none segmenation part 'void'
# NO_OBJ = 61

# pvsg_v1
THING_CLASSES = ['adult', 'baby', 'bag', 'ball', 'ballon', 'basket', 'bed', 'bench', 'bike', 
                 'book', 'bottle', 'bowl', 'box', 'cabinet', 'cake', 'camera', 'candle', 'car', 'cat', 
                 'cellphone', 'chair', 'child', 'cup', 'curtain', 'dog', 'door', 'flower', 'fork', 'fridge', 
                 'glass', 'hat', 'knife', 'light', 'lighter', 'mat', 'paper', 'plant', 'plate', 'rock', 'shelf', 
                 'shoe', 'skateboard', 'sofa', 'spoon', 'stand', 'table', 'towel', 'toy', 'tv', 'window']
STUFF_CLASSES = ['ceiling', 'fence', 'floor', 'grass', 'ground', 'pavement', 'road', 'sand', 'sky', 'slide', 'tree', 'wall', 'water']
BACKGROUND_CLASSES = ['background'] # none segmenation part 'void' - 255
NO_OBJ = 63

NUM_THING = len(THING_CLASSES)
NUM_STUFF = len(STUFF_CLASSES)



def build_classes():
    classes = []
    for cls in THING_CLASSES:
        classes.append(cls)

    for cls in STUFF_CLASSES:
        classes.append(cls)
    return classes


@DATASETS.register_module()
class PVSGSingleVideoImageDataset:
    """
    A dataset only used for test to connect tracker to get query feature tube of a video.
    """
    CLASSES = build_classes()

    def __init__(self,
                 pipeline=None,
                 data_root="./data/pvsg_v1", 
                 video_name="0010_8610561401",  # data.test.video_name
                 test_mode=False,
                 split='test',
                 ):
        assert data_root is not None
        data_root = Path(data_root)
        video_seq_dir = data_root / split

        assert video_seq_dir.exists()
        images_dir = video_seq_dir / "images" / video_name # eg. ./data/pvsg_demo/val/images/0010_8610561401


        # Dataset informartion
        self.num_thing_classes = NUM_THING
        self.num_stuff_classes = NUM_STUFF
        self.num_classes = self.num_thing_classes + self.num_stuff_classes
        assert self.num_classes == len(self.CLASSES)
        self.no_obj_class = NO_OBJ

        img_names = sorted([str(x) for x in (images_dir.rglob("*.png"))])

        # find all images
        images = []
        for frame_id, itm in enumerate(img_names):
            images.append({
                'video_name': video_name,
                'frame_id': frame_id,
                'img': itm,
            })

            assert os.path.exists(images[-1]['img'])
        
        self.images = images # "data" of this dataset
        # self.images = images[:16] # for debug
        # self.reference_images = reference_images
        # self.images = images[104:250] # debug

        # mmdet
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        # misc
        self.flag = self._set_groups()

    def pre_pipelines(self, results):
        results['img_info'] = []
        results['thing_lower'] = 0
        results['thing_upper'] = self.num_thing_classes
        results['ori_filename'] = os.path.basename(results['img'])
        results['filename'] = results['img']


    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        results = copy.deepcopy(self.images[idx]) # eg. [0,0,1]
        self.pre_pipelines(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        results = copy.deepcopy(self.images[idx])
        self.pre_pipelines(results)
        # if not self.ref_seq_index:
        #     results = results[0]
        return self.pipeline(results)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)


    # Copy and Modify from mmdet
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            while True:
                cur_data = self.prepare_train_img(idx)
                if cur_data is None:
                    idx = self._rand_another(idx)
                    continue
                return cur_data

    def __len__(self):
        return len(self.images)

    def _set_groups(self):
        return np.zeros((len(self)), dtype=np.int64)



    def evaluate(
            self,
            results,
            **kwargs
    ): # can only support evaluation in order now !
        max_ins = 10000  # same as self.divisor
        pq_results = []
        pipeline = Compose([
            dict(type='LoadAnnotationsDirect')
        ])
        for idx, _result in enumerate(results):
            img_info = self.images[idx]
            self.pre_pipelines(img_info)
            gt = pipeline(img_info)
            gt_pan = gt['gt_panoptic_seg'].astype(np.int64)
            pan_seg_result = copy.deepcopy(_result['pan_results'])
            pan_seg_map = -1 * np.ones_like(pan_seg_result)
            for itm in np.unique(pan_seg_result):
                if itm >= INSTANCE_OFFSET:
                    cls = itm % INSTANCE_OFFSET
                    ins = itm // INSTANCE_OFFSET
                    pan_seg_map[pan_seg_result == itm] = cls * max_ins + ins
                elif itm == self.num_classes:
                    pan_seg_map[pan_seg_result == itm] = self.num_classes * max_ins
                else:
                    pan_seg_map[pan_seg_result == itm] = itm * max_ins
            assert -1 not in pan_seg_result
            pq_result = vpq_eval([pan_seg_map, gt_pan], num_classes=self.no_obj_class, max_ins=max_ins, ign_id=self.no_obj_class)
            pq_results.append(pq_result)

        iou_per_class = np.stack([result[0] for result in pq_results]).sum(axis=0)[:self.num_classes]
        tp_per_class = np.stack([result[1] for result in pq_results]).sum(axis=0)[:self.num_classes]
        fn_per_class = np.stack([result[2] for result in pq_results]).sum(axis=0)[:self.num_classes]
        fp_per_class = np.stack([result[3] for result in pq_results]).sum(axis=0)[:self.num_classes]
        epsilon = 0.
        sq = iou_per_class / (tp_per_class + epsilon)
        rq = tp_per_class / (tp_per_class + 0.5 * fn_per_class + 0.5 * fp_per_class + epsilon)
        pq = sq * rq
        pq = np.nan_to_num(pq) # set nan to 0.0
        return {
                "PQ": pq,
                "PQ_all": pq.mean(),
                "PQ_th": pq[:self.num_thing_classes].mean(),
                "PQ_st": pq[self.num_thing_classes:].mean(),
                }



            

