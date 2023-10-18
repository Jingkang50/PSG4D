import numpy as np
from pathlib import Path
import torch
import cv2
from torchvision.transforms import transforms as T

from mmdet.core import INSTANCE_OFFSET

class LoadOutputsFromMask2Former:
    """
    Process outputs of all images of a singel video from mask2former to inputs of unitracker.
    Modification from LoadImagesAndMaskObsMOTS: get_data
    """
    def __init__(self,
                 data_cfg, # data_cfg = data.test (in data base config)
                 outputs,
                 tracker_cfg,  # add cfg! (follow unitrack opt)
                 classes):
        data_root = Path(data_cfg.data_root)
        video_folder = data_root / "images" / data_cfg.video_name
        self.num_classes = len(classes)
        self.img_files = sorted([str(x) for x in (video_folder.rglob("*.bmp"))])
        self.transforms = T.Compose([T.ToTensor(), T.Normalize(tracker_cfg.common.im_mean, tracker_cfg.common.im_std)])

        pan_masks_all_images = []
        query_feat_dicts_all_images = []
        for frame_output in outputs:
            pan_masks_all_images.append(frame_output['pan_results'])
            query_feat_dicts_all_images.append(frame_output['query_feats'])
        self.pan_masks_all_images = pan_masks_all_images
        self.query_feat_dicts_all_images = query_feat_dicts_all_images
        
    def _get_binary_masks_and_query_feats(self, pan_mask, query_feat_dict): # for single frame all objects
        object_ids = list(np.unique(pan_mask))
        object_ids.remove(self.num_classes)
        if len(object_ids) == 0:  # deal with the case that no mask was detected from mask2former!!!!
            return np.array([]), []
        assert len(list(query_feat_dict.keys())) == len(object_ids), "Masks and query feats should match!"
    
        binary_masks = []
        query_feats = []
        for object_id in object_ids:
            # binary masks
            obj_binary_mask = (pan_mask == object_id).astype(np.int)
            binary_masks.append(obj_binary_mask)
            # cls ids
            cls_id = object_id % INSTANCE_OFFSET
            # query feat list (map masks order)
            query_feats.append(dict(query_feat=self._unify_query_feat_dim(query_feat_dict[object_id]),
                                    cls_id=cls_id))  # zip the two things together

        return np.stack(binary_masks), query_feats  # obs should be numpy array ! not tensor
    
    def _unify_query_feat_dim(self, query_feat_list):
        if len(query_feat_list) == 1:
            return query_feat_list[0].squeeze() # squeeze (1,256) to (256,)
        else: # get average for stuff
            query_feat_list = [x.squeeze() for x in query_feat_list]
            query_feat_stack = np.stack(query_feat_list)
            return query_feat_stack.mean(axis=0)

    def __getitem__(self, idx):
        img_ori = cv2.imread(self.img_files[idx])
        if img_ori is None:
            raise ValueError('File corrupt {}'.format(self.img_files[idx]))
        # process img: copy from unitrack-video.py
        h, w, _ = img_ori.shape
        img = img_ori
        img = img / 255.
        img = np.ascontiguousarray(img[ :, :, ::-1]) # BGR to RGB
        if self.transforms is not None:
            img = self.transforms(img)
        # process masks and get cls_id
        labels, query_feats = self._get_binary_masks_and_query_feats(self.pan_masks_all_images[idx], 
                                                                     self.query_feat_dicts_all_images[idx])
        return img, labels, img_ori, (h, w), query_feats
        
