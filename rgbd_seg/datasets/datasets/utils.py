from typing import Dict

import numpy as np
import json

from mmdet.core import INSTANCE_OFFSET


class SeqObj:
    """
    This is a seq object class for querying the image for constructing sequence.
    DIVISOR : This divisor is orthogonal with panoptic class-instance divisor (should be large enough).
    """
    DIVISOR = 1000000

    def __init__(self, the_dict: Dict):
        self.dict = the_dict

    def __hash__(self):
        return self.dict['seq_id'] * self.DIVISOR + self.dict['img_id']

    def __eq__(self, other):
        return self.dict['seq_id'] == other.dict['seq_id'] and self.dict['img_id'] == other.dict['img_id']

    def __getitem__(self, attr):
        return self.dict[attr]


def vpq_eval(element, num_classes=61, max_ins=10000, ign_id=61):
    import six
    pred_ids, gt_ids = element
    offset = 1e9
    num_cat = num_classes + 1

    iou_per_class = np.zeros(num_cat, dtype=np.float64)
    tp_per_class = np.zeros(num_cat, dtype=np.float64)
    fn_per_class = np.zeros(num_cat, dtype=np.float64)
    fp_per_class = np.zeros(num_cat, dtype=np.float64)

    def _ids_to_counts(id_array):
        ids, counts = np.unique(id_array, return_counts=True)
        return dict(six.moves.zip(ids, counts))

    pred_areas = _ids_to_counts(pred_ids)
    gt_areas = _ids_to_counts(gt_ids)

    void_id = ign_id * max_ins
    ign_ids = {
        gt_id for gt_id in six.iterkeys(gt_areas)
        if (gt_id // max_ins) == ign_id
    }

    int_ids = gt_ids.astype(np.int64) * offset + pred_ids.astype(np.int64)
    int_areas = _ids_to_counts(int_ids)

    def prediction_void_overlap(pred_id):
        void_int_id = void_id * offset + pred_id
        return int_areas.get(void_int_id, 0)

    def prediction_ignored_overlap(pred_id):
        total_ignored_overlap = 0
        for _ign_id in ign_ids:
            int_id = _ign_id * offset + pred_id
            total_ignored_overlap += int_areas.get(int_id, 0)
        return total_ignored_overlap

    gt_matched = set()
    pred_matched = set()

    for int_id, int_area in six.iteritems(int_areas):
        gt_id = int(int_id // offset)
        gt_cat = int(gt_id // max_ins)
        pred_id = int(int_id % offset)
        pred_cat = int(pred_id // max_ins)
        if gt_cat != pred_cat:
            continue
        union = (
                gt_areas[gt_id] + pred_areas[pred_id] - int_area -
                prediction_void_overlap(pred_id)
        )
        iou = int_area / union
        if iou > 0.5:
            tp_per_class[gt_cat] += 1
            iou_per_class[gt_cat] += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    for gt_id in six.iterkeys(gt_areas):
        if gt_id in gt_matched:
            continue
        cat_id = gt_id // max_ins
        if cat_id == ign_id:
            continue
        fn_per_class[cat_id] += 1

    for pred_id in six.iterkeys(pred_areas):
        if pred_id in pred_matched:
            continue
        if (prediction_ignored_overlap(pred_id) / pred_areas[pred_id]) > 0.5:
            continue
        cat = pred_id // max_ins
        fp_per_class[cat] += 1

    return iou_per_class, tp_per_class, fn_per_class, fp_per_class


def pan_mm2hb(pred_pan_map, num_classes, divisor=10000):
    pan_seg_map = - np.ones_like(pred_pan_map)
    for itm in np.unique(pred_pan_map):
        if itm >= INSTANCE_OFFSET:
            cls = itm % INSTANCE_OFFSET
            ins = itm // INSTANCE_OFFSET
            pan_seg_map[pred_pan_map == itm] = cls * divisor + ins
        elif itm == num_classes:
            pan_seg_map[pred_pan_map == itm] = num_classes * divisor
        else:
            pan_seg_map[pred_pan_map == itm] = itm * divisor
    assert -1 not in pan_seg_map
    return pan_seg_map


class PVSGAnnotation:
    def __init__(self, anno_file, video_ids):
        with open(anno_file, 'r') as f:
            anno = json.load(f)
        self.anno = anno['data']

        videos = {}
        for video_anno in self.anno:
            if video_anno['video_id'] in video_ids:
                videos[video_anno['video_id']] = video_anno

        self.videos = videos

    def __getitem__(self, vid):
        assert vid in self.videos
        return self.videos[vid]
        