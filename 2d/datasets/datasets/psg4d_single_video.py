import os
from pathlib import Path

import copy

import numpy as np

from mmdet.core import INSTANCE_OFFSET
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

# from datasets.datasets.builder import DATASETS
from datasets.datasets.utils import SeqObj, PVSGAnnotation, vpq_eval

# GTA
THING_CLASSES = ['person', 'car', 'motorcycle', 'truck', 'bird', 'dog', 'handbag', 
                 'suitcase', 'bottle', 'cup', 'bowl', 'chair', 'potted plant', 'bed', 
                 'dining table', 'tv', 'laptop', 'cell phone', 'bag', 'bin', 'box', 
                 'door', 'road barrier', 'stick', 'chair', 'sofa', 'table', 'bed']
STUFF_CLASSES = ['sky', 'floor', 'ceiling', 'ground', 'wall', 'grass', 'fence', 'stair', 'window']
BACKGROUND_CLASSES = ['background']
NO_OBJ = 37

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
class PSG4DSingleVideoDataset:
    CLASSES = build_classes()
    def __init__(self,
                 pipeline=None,
                 data_root="./data/demo", 
                 # video_name="ah_3b_mcs_5",
                 video_name="demo",
                 test_mode=True,
                 split='val',
                 ):
        assert data_root is not None
        data_root = Path(data_root)
        # video_seq_dir = data_root / split
        video_seq_dir = data_root # TODO - temporal change

        assert video_seq_dir.exists()
        images_dir = video_seq_dir / "images" / video_name

        # Dataset informartion
        self.num_thing_classes = NUM_THING
        self.num_stuff_classes = NUM_STUFF
        self.num_classes = self.num_thing_classes + self.num_stuff_classes
        assert self.num_classes == len(self.CLASSES)
        self.no_obj_class = NO_OBJ

        img_names = sorted([str(x) for x in (images_dir.rglob("*.bmp"))])

        # find all images
        images = []
        for frame_id, itm in enumerate(img_names):
            images.append({
                'video_name': video_name,
                'frame_id': frame_id,
                'img': itm,
                'dep_img': itm.replace('images', 'depth').replace("bmp", "npy"),
                #'dep_img': itm.replace('images', 'depth').replace("bmp", "png"), # for demo
            })

            assert os.path.exists(images[-1]['img'])
            assert os.path.exists(images[-1]['dep_img'])

        self.images = images # "data" of this dataset
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
        max_ins = 10000
        ap_results = []

        pipeline = Compose([dict(type='LoadAnnotationsDirect')])

        for idx, _result in enumerate(results):
            img_info = self.images[idx]
            self.pre_pipelines(img_info)
            print(img_info)
            gt = pipeline(img_info)
            
            # Your method for extracting ground truth labels and prediction results
            gt_labels, pred_labels, pred_scores = extract_labels_and_scores(gt, _result)
            
            # Compute true and false positives for each threshold
            thresholds = np.sort(pred_scores)[::-1]
            tp = np.zeros_like(thresholds)
            fp = np.zeros_like(thresholds)
            for i, threshold in enumerate(thresholds):
                selected = pred_scores >= threshold
                matched = np.isin(pred_labels[selected], gt_labels)
                tp[i] = np.sum(matched)
                fp[i] = np.sum(~matched)
            
            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps)
            recalls = tp_cumsum / len(gt_labels)
            
            # Compute average precision (AP)
            precisions_interp = np.maximum.accumulate(precisions[::-1])[::-1]
            recalls_diff = np.diff(recalls, prepend=0)
            ap = np.sum(precisions_interp * recalls_diff)
            
            ap_results.append(ap)
        
        mAP = np.mean(ap_results)
        return {"mAP": mAP}
