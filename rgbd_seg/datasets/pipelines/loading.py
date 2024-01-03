from typing_extensions import Literal
from unicodedata import category

import mmcv
import numpy as np
from PIL import Image
from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile


def bitmasks2bboxes(bitmasks):
    bitmasks_array = bitmasks.masks
    boxes = np.zeros((bitmasks_array.shape[0], 4), dtype=np.float32)
    x_any = np.any(bitmasks_array, axis=1)
    y_any = np.any(bitmasks_array, axis=2)
    for idx in range(bitmasks_array.shape[0]):
        x = np.where(x_any[idx, :])[0]
        y = np.where(y_any[idx, :])[0]
        if len(x) > 0 and len(y) > 0:
            boxes[idx, :] = np.array((x[0], y[0], x[-1], y[-1]), dtype=np.float32)
    return boxes


@PIPELINES.register_module()
class LoadImgDirect:
    """Go ahead and just load image
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 with_depth=False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.with_depth = with_depth

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict requires "img" which is the img path.

        Returns:
            dict: The dict contains loaded image and meta information.
            'img' : img
            'img_shape' : img_shape
            'ori_shape' : original shape
            'img_fields' : the img fields
        """
        img = mmcv.imread(results['img'], channel_order='rgb', flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        if self.with_depth:
            dep_img = np.load(results['dep_img'])
            # dep_img = mmcv.imread(results['dep_img'], channel_order='rgb', flag=self.color_type) # for demo
            # copy 3 times along channel dimension
            dep_img = np.repeat(dep_img[:, :, np.newaxis], 3, axis=2) # TODO - check how to do it!
            results['dep_img'] = dep_img
            
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        if self.with_depth:
            results['img_fields'] = ['img', 'dep_img'] # !!!
        else:
            results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', ")
        return repr_str

@PIPELINES.register_module()
class LoadMultiImagesDirect(LoadImgDirect):
    """Load multi images from file.
    Please refer to `mmdet.datasets.pipelines.loading.py:LoadImageFromFile`
    for detailed docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.
        For each dict in `results`, call the call function of
        `LoadImageFromFile` to load image.
        Args:
            results (list[dict]): List of dict from
                :obj:`mmtrack.CocoVideoDataset`.
        Returns:
            list[dict]: List of dict that contains loaded image.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs

@PIPELINES.register_module()
class LoadAnnotationsDirect:
    """New version of VPS dataloader
        PVSG dataset.
    """

    def __init__(
            self,
            with_relation=False,
            divisor: int = 10000,
            instance_only: bool = False,
            with_ps_id: bool = False,
    ):
        self.divisor = divisor
        self.is_instance_only = instance_only
        self.with_ps_id = with_ps_id # do we need this?

    def __call__(self, results):
        ann_file = results['ann']
        if ann_file.lower().endswith('.npy'):
            pan_mask = np.load(ann_file)
        else:
            pan_mask = np.array(Image.open(ann_file)).astype(np.int64) # palette format saved one-channel image
        # default:int16, need to change to int64 to avoid data overflow
        objects_info = results['objects']
        cates2id = results['pre_hook']

        gt_semantic_seg = -1 * np.ones_like(pan_mask)
        classes = []
        masks = []
        instance_ids = []
        for instance_id in np.unique(pan_mask): # 0,1...n object id
            # filter background (void) class
            if instance_id == 0: # no segmentation area
                category = 'background'
                gt_semantic_seg[pan_mask == instance_id] = cates2id(category) # 61
            else: # gt_label & gt_masks do not include "void"
                for _object in objects_info:
                    if _object['object_id'] == instance_id:
                        category = _object['category']
                        break
                # category = objects_info[instance_id - 1]['category'] # TODO - TEMPORAL change for gta json data
                semantic_id = cates2id(category)
                gt_semantic_seg[pan_mask == instance_id] = semantic_id
                classes.append(semantic_id)
                instance_ids.append(instance_id)
                masks.append((pan_mask == instance_id).astype(np.int))
        
        # check semantic mask
        gt_semantic_seg = gt_semantic_seg.astype(np.int64)
        assert -1 not in np.unique(gt_semantic_seg).astype(np.int)
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'] = ['gt_semantic_seg']

        # add panoptic_seg in "vps encoded format" for evaluation use --------------------------------------
        ps_id = gt_semantic_seg * self.divisor + pan_mask
        results['gt_panoptic_seg'] = ps_id
        # --------------------------------------------------------------------------------------------------
        
        if len(classes) == 0: # this image is annotated as "all background", no classes, no masks... (very few images)
            print("{} is annotated as all background!".format(results['filename']))
            gt_labels = np.array(classes).astype(np.int) # empty array
            gt_instance_ids = np.array(instance_ids).astype(np.int)
            _height, _width = pan_mask.shape
            gt_masks = BitmapMasks(masks, height=_height, width=_width)
        else:
            gt_labels = np.stack(classes).astype(np.int)
            gt_instance_ids = np.stack(instance_ids).astype(np.int)
            _height, _width = pan_mask.shape
            gt_masks = BitmapMasks(masks, height=_height, width=_width)

            # check the sanity of gt_masks
            verify = np.sum(gt_masks.masks.astype(np.int), axis=0)
            assert (verify == (pan_mask != 0).astype(verify.dtype)).all() # none-background area exactly same

            # for instance only -- might not use
            if self.is_instance_only:
                gt_masks.masks = np.delete(
                    gt_masks.masks,
                    gt_labels >= results['thing_upper'],
                    axis=0
                )
                gt_instance_ids = np.delete(
                    gt_instance_ids,
                    gt_labels >= results['thing_upper'],
                )
                gt_labels = np.delete(
                    gt_labels,
                    gt_labels >= results['thing_upper'],
                )
        
        results['gt_labels'] = gt_labels
        results['gt_masks'] = gt_masks
        results['gt_instance_ids'] = gt_instance_ids  # ??
        results['mask_fields'] = ['gt_masks']

        # generate boxes
        boxes = bitmasks2bboxes(gt_masks)
        results['gt_bboxes'] = boxes
        results['bbox_fields'] = ['gt_bboxes']
        return results

@PIPELINES.register_module()
class LoadMultiAnnotationsDirect(LoadAnnotationsDirect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if _results is None:
                return None
            outs.append(_results)
        return outs



