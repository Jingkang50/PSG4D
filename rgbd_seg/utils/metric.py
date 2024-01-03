from collections import defaultdict
from importlib_metadata import re
import numpy as np
import pickle
import json
import os
from tqdm import tqdm

class Result(dict):
    def __init__(self, 
                 vid,
                 bbox_trajs=None,
                 seg_masks=None,
                 relations=None,
                 *args, **kwargs):
        super(Result, self).__init__(*args, **kwargs)
        self.vid = vid
        self.bbox_trajs = bbox_trajs
        self.seg_masks = seg_masks
        self.relations = relations

        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v


    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Result, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Result, self).__delitem__(key)
        del self.__dict__[key]




def generate_helper_relation_indicator(num_frames, durs):
    # generate a true/false 1d-array to indicate where has relation
    relation_indicator = np.zeros((num_frames,), dtype=bool)
    for dur in durs:
        start_idx, end_idx = dur[0]-1, dur[1]-1 
        relation_indicator[start_idx: end_idx + 1] = True
    return relation_indicator

def compute_bboxes_intersection(bboxes1, bboxes2):
    # bboxes: [k, 4]
    assert bboxes1.shape == bboxes2.shape
    tot_intersection = 0
    for bbox1, bbox2 in zip(bboxes1, bboxes2):
        left = max(bbox1[0], bbox2[0])
        top = max(bbox1[1], bbox2[1])
        right = min(bbox1[2], bbox2[2])
        bottom = min(bbox1[3], bbox2[3])
        tot_intersection += max(0, right - left + 1) * max(0, bottom - top + 1)
    return tot_intersection

def compute_bboxes_union(bboxes):
    # bboxes: [k, 4]
    union = 0
    for bbox in bboxes:
        union += (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    return union

def compute_masks_intersection(masks1, masks2):
    # masks: [k, w, h]
    assert masks1.shape == masks2.shape
    tot_intersection = 0
    for mask1, mask2 in zip(masks1, masks2):
        tot_intersection += np.count_nonzero(np.logical_and(mask1, mask2))
    return tot_intersection

def compute_masks_union(masks):
    # masks: [k, w, h]
    union = 0
    for mask in masks:
        union += np.count_nonzero(mask)
    return union

def viou_over_whole_length_video(tube1, durs1, 
                                 tube2, durs2, 
                                 detection_method='pan_seg'):
    # tube - bbox tube or mask_tube
    # traj - [num_frames, 4]; durs - list of frame_id intervals
    # Note: frame_id starts from 1
    tot_num_frames = tube1.shape[0]
    tube1_has_relation = generate_helper_relation_indicator(tot_num_frames, durs1)
    tube2_has_relation = generate_helper_relation_indicator(tot_num_frames, durs2)
    
    # intersection for relation
    has_relation_overlap = np.logical_and(tube1_has_relation, tube2_has_relation) # True indicates framse in the intersection of "having relation"
    tube1_has_relation_overlap = tube1[has_relation_overlap] # the bboxes in the intersection of relation
    tube2_has_relation_overlap = tube2[has_relation_overlap] # for bbox: [num_frames_intersection, 4]' mask: [num_frames_interset, w, h]
    
    # intersection for V
    if detection_method == 'bbox':
        v_intersection = compute_bboxes_intersection(tube1_has_relation_overlap, tube2_has_relation_overlap)
        # union for V
        tube1_v_union = compute_bboxes_union(tube1[tube1_has_relation])
        tube2_v_union = compute_bboxes_union(tube2[tube2_has_relation])
    else:
        v_intersection = compute_masks_intersection(tube1_has_relation_overlap, tube2_has_relation_overlap)
        # union for V
        tube1_v_union = compute_masks_union(tube1[tube1_has_relation])
        tube2_v_union = compute_masks_union(tube2[tube2_has_relation])
    
    v_union = tube1_v_union + tube2_v_union - v_intersection
    if v_union == 0:
        return 0.0
    return float(v_intersection / v_union)
    

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).

    Adopted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_relations_of_singel_video(groundtruth, 
                                   prediction, 
                                   viou_theshhold=0.5, 
                                   detection_method='pan_seg'):
    # detection_method = [bbox, pan_seg]
    assert detection_method in ['bbox', 'pan_seg'], "Invalid detection method! Choose from [bbox, pan_seg]"
    if detection_method == 'pan_seg':
        # load inputs
        gt_mask_tubes = groundtruth.seg_masks
        gt_relations = groundtruth.relations
        pred_mask_tubes = prediction.seg_masks
        pred_relations = prediction.relations

        pred_relations.sort(key=lambda x: x['score'], reverse=True)
        gt_detected = np.zeros((len(gt_relations),), dtype=bool) # if this gt relation is matched
        hit_scores = np.ones((len(pred_relations))) * -np.inf

        for pred_idx, pred_relation in enumerate(pred_relations):
            ov_max = -float('Inf')
            k_max = -1
            pred_triplet = pred_relation['triplet']  # ((597, 54), (599, 21), 12)
            pred_cls_triplet = (pred_triplet[0][1], pred_triplet[1][1], pred_triplet[2]) # (54, 21, 12)
            pred_s_id, pred_o_id = pred_triplet[0][0], pred_triplet[1][0] # (597, 599, 12)
            pred_durs = pred_relation['durs']
            # iterate over all gt relations to find one that matches the pred_relation most
            for gt_idx, gt_relation in enumerate(gt_relations):
                gt_triplet = gt_relation['triplet']
                gt_cls_triplet = (gt_triplet[0][1], gt_triplet[1][1], gt_triplet[2])
                if not gt_detected[gt_idx] and pred_cls_triplet == gt_cls_triplet:
                    gt_s_id, gt_o_id = gt_triplet[0][0], gt_triplet[1][0]
                    # get trajs of object of this id (whole length video - per frame detection results)
                    gt_subject_mask_tubes = gt_mask_tubes[gt_s_id]
                    pred_subject_mask_tubes = pred_mask_tubes[pred_s_id]
                    gt_object_mask_tubes = gt_mask_tubes[gt_o_id]
                    pred_object_mask_tubes = pred_mask_tubes[pred_o_id]
                    # get relation durs
                    gt_durs = gt_relation['durs']
                    # compute vIoU over the whole-length video
                    subject_viou = viou_over_whole_length_video(gt_subject_mask_tubes, gt_durs, 
                                                                pred_subject_mask_tubes, pred_durs,
                                                                detection_method)
                    object_viou = viou_over_whole_length_video(gt_object_mask_tubes, gt_durs, 
                                                               pred_object_mask_tubes, pred_durs,
                                                               detection_method)
                    ov = min(subject_viou, object_viou)
                    if ov >= viou_theshhold and ov > ov_max:
                        ov_max = ov
                        k_max = gt_idx
            
            if k_max >= 0:
                hit_scores[pred_idx] = pred_relation['score'] # finally, if no gt match, the score for this pred will remain -inf
                gt_detected[k_max] = True # this gt is finished mapping
        tp = np.isfinite(hit_scores)
        fp = ~tp
        cum_tp = np.cumsum(tp).astype(np.float32)
        cum_fp = np.cumsum(fp).astype(np.float32)
        rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
        return prec, rec, hit_scores
        



def evaluate(groundtruths, 
             predictions, 
             viou_threshhold=0.5, 
             topK_nreturns=[10, 20, 50], 
             detection_method='pan_seg'):
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)

    tot_gt_relations = 0
    print('Computing average precision MAP over {} videos...'.format(len(groundtruths)))
    # iterate over all videos
    for groundtruth, prediction in zip(groundtruths, predictions):
        # skip if no gt relations
        if len(groundtruth.relations) == 0:
            continue
        vid = groundtruth.vid
        tot_gt_relations += len(groundtruth.relations) # for total number of gt_relation_intances
        vid_prec, vid_rec, vid_rel_scores = eval_relations_of_singel_video(groundtruth, 
                                                                           prediction, 
                                                                           viou_theshhold=viou_threshhold, 
                                                                           detection_method=detection_method)

        video_ap[vid] = voc_ap(vid_rec, vid_prec)
        tp = np.isfinite(vid_rel_scores)
        for nre in topK_nreturns:
            cut_off = min(nre, vid_rel_scores.size)  # det_scores.size -> number of pred_relations
            tot_scores[nre].append(vid_rel_scores[:cut_off])  
            # 每个video自己cut off, scores在之前以及sort过了，cut off就是分数最高的前k个，但是分数高也可能一个gt都没有匹配到，是false
            tot_tp[nre].append(tp[:cut_off])
    
    # calculate mean ap for detection
    mean_ap = np.mean(list(video_ap.values()))  # mean_ap -> average on video level
    # calculate recall for detection
    rec_at_n = dict()
    for nre in topK_nreturns:
        scores = np.concatenate(tot_scores[nre])  # video level拆成relation_instances level
        tps = np.concatenate(tot_tp[nre])   # video level拆成relation_instances level
        sort_indices = np.argsort(scores)[::-1]  # 所有relation instance sort
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        rec_at_n[nre] = rec[-1]
    return mean_ap, rec_at_n



if __name__ == "__main__":
    with open("/mnt/lustre/wxpeng/OpenPVSG/data/pvsg_v1_clean/val/val_video_names.json", "r") as f:
        vids = json.load(f)
    with open("/mnt/lustre/wxpeng/OpenPVSG/data/metric_input3/gt/data.pickle", "rb") as f:
        gt_all_vids = pickle.load(f)
    pred_result_root = "/mnt/lustre/wxpeng/OpenPVSG/data/metric_input_ckpt50/transformer"
    groundthruths = []
    predictions = []
    # vids = vids[2:4]
    for vid in tqdm(vids):
        gt_data = gt_all_vids[vid]
        pred_path = os.path.join(pred_result_root, f"{vid}.pickle")
        with open(pred_path, "rb") as f:
            pred_data = pickle.load(f)
        for x in pred_data['all_pair_clips']:
            x['score'] = 1
        groundtruth = Result(vid=vid,
                            seg_masks=gt_data['gt_mask'],
                            relations=gt_data['gt_relation'])
        prediction = Result(vid=vid,
                            seg_masks=pred_data['pred_masks'],
                            relations=pred_data['all_pair_clips'])
        groundthruths.append(groundtruth)
        predictions.append(prediction)

    mean_ap, recal_atn = evaluate(groundthruths, 
                              predictions, 
                              viou_threshhold=0.1, 
                              topK_nreturns=[10, 20, 50], 
                              detection_method='pan_seg') 
    print("mean_ap:")
    print(mean_ap)
    print("recall at n")
    print(recal_atn)
