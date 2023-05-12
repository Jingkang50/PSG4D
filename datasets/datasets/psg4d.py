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
                 'door', 'road barrier', 'stick']
STUFF_CLASSES = []
BACKGROUND_CLASSES = ['background']
NO_OBJ = 24

NUM_THING = len(THING_CLASSES)
NUM_STUFF = len(STUFF_CLASSES)


def cates2id(category):
    # if category == 'background':
    #     return 1
    CLASSES = THING_CLASSES + STUFF_CLASSES + BACKGROUND_CLASSES # void class will be indexed 61 manually
    class2ids = dict(zip(CLASSES, range(len(CLASSES))))
    return class2ids[category]


def build_classes():
    classes = []
    for cls in THING_CLASSES:
        classes.append(cls)

    for cls in STUFF_CLASSES:
        classes.append(cls)
    return classes



@DATASETS.register_module()
class PSG4DDataset:
    # CLASSES = build_classes()
    def __init__(self,
                 pipeline=None,
                 data_root="./data/GAT", 
                 annotation_file="psg4d_val_v1.json",
                 test_mode=False,
                 split='train',
                 with_relation: bool = False
                 ):
        assert data_root is not None
        data_root = Path(data_root)
        anno_file = data_root / annotation_file
        video_seq_dir = data_root / split

        assert anno_file.exists()
        assert video_seq_dir.exists()
        images_dir = video_seq_dir / "images"

        Dataset informartion
        self.num_thing_classes = NUM_THING
        self.num_stuff_classes = NUM_STUFF
        self.num_classes = self.num_thing_classes + self.num_stuff_classes
        assert self.num_classes == len(self.CLASSES)
        self.no_obj_class = NO_OBJ

        img_names = sorted([str(x) for x in (images_dir.rglob("*.bmp"))])

        anno = PVSGAnnotation(anno_file) # TODO - to implement PSG4DAnnotation
        # find all images
        images = []
        ref_images = []
        vid2seq_id = {}
        seq_count = 0
        for itm in img_names:
            tokens = itm.split(sep="/")
            vid, img_id = tokens[-2], tokens[-1].split(sep=".")[0]
            vid_anno = anno[vid]  # annotation_dict of this video

            # map vid to seq_id (seq_id starts from 0)
            if vid in vid2seq_id:
                seq_id = vid2seq_id[vid]
            else:
                seq_id = seq_count
                vid2seq_id[vid] = seq_count
                seq_count += 1
            
            images.append({
                'rgb_img': itm,
                'dep_img': itm.replace('images', 'depth').replace("bmp", "npy"),
                'ann': itm.replace('images', 'visible').replace("bmp", "npy"),
                'objects': vid_anno['objects'],
                'pre_hook': cates2id,  
            })

            assert os.path.exists(images[-1]['rgb_img'])
            assert os.path.exists(images[-1]['dep_img'])
            assert os.path.exists(images[-1]['ann'])

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
        results['ori_filename'] = os.path.basename(results['rgb_img'])
        results['filename'] = results['rgb_img']

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
        pass
