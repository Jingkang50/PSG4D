import os, glob, json
from pathlib import Path

import copy

import numpy as np

from mmdet.core import INSTANCE_OFFSET
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

from datasets.datasets.utils import SeqObj, PVSGAnnotation, vpq_eval


@DATASETS.register_module()
class GTASingleVideoDataset:
    def __init__(self,
                 pipeline=None,
                 data_root="../data/sailvos3d", 
                 annotation_file="sailvos3d.json",
                 video_name="ah_3b_mcs_5",
                 test_mode=True,
                 split='val',
                 ):
    def __init__(
        self,
        pipeline=None,
        data_root='./data/',
        annotation_file='pvsg.json',
        video_name="ah_3b_mcs_5",
        test_mode=False,
        split='test',
    ):
        assert data_root is not None
        data_root = Path(data_root)
        anno_file = data_root / annotation_file

        with open(anno_file, 'r') as f:
            anno = json.load(f)

        # collect class names
        self.THING_CLASSES = anno['objects']['thing']  # 26
        self.STUFF_CLASSES = anno['objects']['stuff']  # 9
        self.BACKGROUND_CLASSES = ['background']
        self.CLASSES = self.THING_CLASSES + self.STUFF_CLASSES
        self.num_thing_classes = len(self.THING_CLASSES)
        self.num_stuff_classes = len(self.STUFF_CLASSES)
        self.num_classes = len(self.CLASSES)  # 126

        # collect video id within the split
        video_ids, img_names = [], []
        for video_id in anno['split'][split]:
            video_ids.append(video_id)
            img_names += glob.glob(
                os.path.join(data_root, 'images', video_id, '*.bmp'))
        assert anno_file.exists()
        assert data_root.exists()

        # get annotation file
        anno = PVSGAnnotation(anno_file, video_ids)

        # find all images
        images = []
        for itm in img_names:
            vid = itm.split(sep='/')[-2]
            vid_anno = anno[vid]

            images.append({
                'img': itm,
                'dep_img': itm.replace('images', 'depth').replace("bmp", "npy"),
                'ann': itm.replace('images', 'visible').replace("bmp", "npy"),
                'objects': vid_anno['object'],
                'pre_hook': self.cates2id,
            })

            assert os.path.exists(images[-1]['img'])
            assert os.path.exists(images[-1]['dep_img'])
            assert os.path.exists(images[-1]['ann'])

        self.images = images
        # self.images = images[:16] # for debug
        
        # mmdet
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        # misc
        self.flag = self._set_groups()

    def cates2id(self, category):
        class2ids = dict(
            zip(self.CLASSES + self.BACKGROUND_CLASSES,
                range(len(self.CLASSES + self.BACKGROUND_CLASSES))))
        return class2ids[category]

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
