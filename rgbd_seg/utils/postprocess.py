import numpy as np
from itertools import groupby
from pathlib import Path
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import json
import pickle
import pycocotools.mask as mask_utils

from ..datasets.datasets.utils import PVSGAnnotation

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)



# ---------------------------------------------- load pred masks --------------------------------------------------
def load_pred_mask_tubes(label_path):
    # label_path: mask txt output from unitrack
    print("Loading pred mask tubes...", flush=True)
    labels = []
    results = []
    with open(label_path, 'r') as f:
        for line in f:
            labels.append(line.strip().split())

    for label in tqdm(labels):
        frame_id, track_id, _, h, w, m = label
        rle = {'size':(int(h),int(w)), 'counts':m}
        mask = mask_utils.decode(rle)
        results.append(dict(fid=frame_id, tid=track_id, mask=mask))
    
    def key_func(k):
        return k['tid']
    
    # sort data by 'tid' key.
    results = sorted(results, key=key_func)
    # group by tid
    masks_grp_by_tid = {}
    for key, value in groupby(results, key_func):
        masks_grp_by_tid[key] = list(value)
    return masks_grp_by_tid

# ------------------------------------------------- load gt masks ----------------------------------------------------
def read_pan_mask_next_frame(mask_tubes, pan_mask, cur_len, h, w):
    cur_ids = list(mask_tubes.keys())
    new_ids = list(np.unique(pan_mask))
    all_ids = list(set(cur_ids + new_ids))
    # for no mask frame -- all zeros
    dummy_mask = np.zeros((h,w))

    for instance_id in all_ids:
        if instance_id == 0:
            continue
        if instance_id not in new_ids: # this frame has no this object
            mask_tubes[instance_id].append(dummy_mask)
            continue
        if instance_id not in cur_ids: # this object first show up
            mask_tubes[instance_id].extend([dummy_mask for i in range(cur_len)])
            
        mask_tubes[instance_id].append((pan_mask == instance_id).astype(int))
    return mask_tubes

def load_gt_mask_tubes(gt_mask_path):
    print("Loading gt mask tubes...", flush=True)
    gt_pan_mask_paths = [str(x) for x in sorted(gt_mask_path.rglob("*.png"))]
    
    mask_tubes = defaultdict(list)
    cur_len = 0
    for mask_path in tqdm(gt_pan_mask_paths):
        pan_mask = np.array(Image.open(mask_path))
        h, w = pan_mask.shape
        mask_tubes = read_pan_mask_next_frame(mask_tubes, pan_mask, cur_len, h, w)
        cur_len += 1
    return mask_tubes


def check_has_none(mask_tube):
    return any(ele is None for ele in mask_tube)

# ---------------------------------------- match gt mask tube and pred mask tube -------------------------------------------------
def binaryMaskIOU(mask1, mask2):   
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1==1, mask2==1))
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou

def match_gt_pred_mask_tubes(pred_mask_tubes, gt_mask_tubes):
    print("Matching pred mask tubes with gt mask tubes...", flush=True)
    # init iou score for all pred mask tube
    assigned_labels = {tid: -1 for tid in pred_mask_tubes.keys()}
    iou_scores = {tid: -1 for tid in pred_mask_tubes.keys()}

    iou_thres = 0.85
    for tid, pred_mask_tube in tqdm(pred_mask_tubes.items()): # iterare all pred mask tubes
        gt_id = -1 # pred mask tube只会对应一个gt tube，但是可能对应的是part of it
        iou_score = -np.inf
        # compute viou with all gt mask tubes
        for gt_instance_id, gt_mask_tube in gt_mask_tubes.items():
            viou = 0.0
            count = 0.0
            for pred_mask_dict in pred_mask_tube: # iterate every frame in a pred mask tube
                fid = int(pred_mask_dict['fid']) - 1 # our gt starts from 0, but pred starts from 1, so need - 1
                pred_mask = pred_mask_dict['mask']
                gt_mask = gt_mask_tube[fid]
                iou = binaryMaskIOU(pred_mask, gt_mask)
                viou += iou
                count += 1
            viou = viou / count
            if viou >= iou_thres and viou > iou_score:
                gt_id = gt_instance_id
                iou_score = viou
        if gt_id != -1: # has some > 0.85
            iou_scores[tid] = iou_score # remember this highest iou score for this pred mask tube
            assigned_labels[tid] = gt_id # remember which part of gt is assigned to a pred(tid) - 这条gt_tube的dummy_id的部分被assign给当前pred mask tube

    return assigned_labels, iou_scores
        
       
# ----------------------- assign labels to qf tube pair (match qf and gt relation) ----------------------------------------------
# get filtered qf_tube out
def filter_qf_tubes(assigned_labels, qf_tube_obj_list):
    # take out those have assigned labels in gt
    qf_tubes_filtered = []
    for (tid, gt_id), qf_tube_obj in zip(assigned_labels.items(), qf_tube_obj_list):
        if gt_id == -1:
            continue
        qf_tubes_filtered.append(dict(tid=int(tid),
                                      gt_id=gt_id,
                                      qf_tube=qf_tube_obj.qf_tube))
    return qf_tubes_filtered

def pair_qf_tubes(qf_tubes_filtered):
    # pair to get true/false tube to indicate which part should have a label (both have qf tubes)
    # avoid pairing with itself (same gt_id)
    pairs = []
    for s in qf_tubes_filtered:
        for o in qf_tubes_filtered:
            if (s['tid'] == o['tid']) or (s['gt_id'] == o['gt_id']):
                continue
            s_indicator = np.array([True if ele is not None else False for ele in s['qf_tube']])
            o_indicator = np.array([True if ele is not None else False for ele in o['qf_tube']])
            pair_indicator = s_indicator * o_indicator
            # also get tubes here
            s_qf_tube = [x['query_feat'] if x is not None else None for x in s['qf_tube']]
            o_qf_tube = [x['query_feat'] if x is not None else None for x in o['qf_tube']]
            so_qf_tubes = [s_qf_tube, o_qf_tube]
            pairs.append(dict(so_tid=[s['tid'], o['tid']],
                              so_gt_id=[s['gt_id'], o['gt_id']],
                              so_qf_tubes=so_qf_tubes,
                              indicator=pair_indicator))
    return pairs
    

def assign_relation_label(gt_relations, pairs):
    # assigne label (label list) to every qf_tube pair in pairs
    print("Assigning labels to every qf tube pair...", flush=True)
    if len(pairs) == 0:
        return [], []
    
    labels = []
    pairs_filtered = []
    num_frames = len(pairs[0]['indicator'])
    for pair in tqdm(pairs):
        label_this_pair = [[] for i in range(num_frames)]
        s_gt_id, o_gt_id = pair['so_gt_id']
        indicator = pair['indicator']
        has_rel = False
        for gt_relation in gt_relations:
            if s_gt_id == gt_relation[0] and o_gt_id == gt_relation[1]:
                predicate = gt_relation[2]
                intervals = gt_relation[3]
                for interval in intervals: # one might have sevel time intervals for a relation
                    # remember our interval is not close on the right side but "range()" does! need + 1
                    start, end = interval[0], interval[1]
                    for i in range(start, end + 1):
                        if indicator[i]: # if also has a pair tube here
                            has_rel = True
                            label_this_pair[i].append(predicate)
        if has_rel:
            labels.append(dict(so_tid=pair['so_tid'],
                               so_gt_id=pair['so_gt_id'],
                               label=label_this_pair))
            pairs_filtered.append(pair['so_qf_tubes'])
    return pairs_filtered, labels
    
def get_labels_single_video(assigned_labels, qf_tube_obj_list, gt_relations):
    qf_tubes_filtered = filter_qf_tubes(assigned_labels, qf_tube_obj_list)
    pairs = pair_qf_tubes(qf_tubes_filtered)
    pairs_filtered, labels = assign_relation_label(gt_relations, pairs)
    return pairs_filtered, labels

# ---------------------------------------------- main ---------------------------------------------------------------------
# save_root = Path("/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image/work_dirs/test_default_ckpt4_tracking_and_save_qf2")
# gt_masks_root = Path("/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image/data/pvsg_demo/train/masks") 
# gt = PVSGAnnotation("/mnt/lustre/jkyang/wxpeng/CVPR23/pvsg_data/pvsg_demo.json")
# def postprocess_resutls_assign_labels_to_qf_tubes_single_video(vid):
#     pred_mask_path = save_root / vid / "quantitive/masks.txt"
#     pred_mask_tubes = load_pred_mask_tubes(pred_mask_path)
    
#     gt_mask_path = gt_masks_root / vid
#     gt_mask_tubes = load_gt_mask_tubes(gt_mask_path)
    
#     assigned_labels, iou_scores = match_gt_pred_mask_tubes(pred_mask_tubes, gt_mask_tubes)
    
#     qf_tube_obj_list = load_pickle(save_root / vid / "query_feats.pickle")
#     gt_relations = gt[vid]['relations']
#     pairs_filtered, labels = get_labels_single_video(assigned_labels, qf_tube_obj_list, gt_relations)
#     return pairs_filtered, labels