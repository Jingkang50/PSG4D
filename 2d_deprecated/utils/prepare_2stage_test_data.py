import os
import os.path as osp
from pathlib import Path
from PIL import Image
import json
import pickle
from tqdm import tqdm
import numpy as np
from itertools import groupby
from collections import defaultdict
import pycocotools.mask as mask_utils
import sys
sys.path.append("/mnt/lustre/wxpeng/OpenPVSG")


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)
    
class PVSGAnnotation:
    def __init__(self, anno_file):
        with open(anno_file, "r") as f:
            anno = json.load(f)
            
        self.anno = anno
        videos = {}
        for video_anno in anno:
            videos[video_anno['video_id']] = video_anno
        self.videos = videos


    def __getitem__(self, vid):
        assert vid in self.videos
        return self.videos[vid]

# global var
result_root = "/mnt/lustre/wxpeng/OpenPVSG/work_dirs/test_full_val8"  # openpvsg ckpt7 val results (19 videos)
vids = sorted(os.listdir(result_root))
# gt
gt = PVSGAnnotation("/mnt/lustre/wxpeng/OpenPVSG/data/pvsg_v1_clean/val/pvsg_clean.json")

############################################################### qf ###############################################################

# summary - given qf_path, output
def valid_tube(qft):
    return qft.end_frame_id - qft.start_frame_id >=3

def can_pair(qft1, qft2):
    # has time overlap
    start1, end1 = qft1.start_frame_id, qft1.end_frame_id
    start2, end2 = qft2.start_frame_id, qft2.end_frame_id
    return not (end1 <= start2 or end2 <= start1) # 不取=，只有一帧重合也不pair
    
def generate_tube_data(qft):
    qf_feats_tube = np.array([x['query_feat'] if x is not None else None for x in qft.qf_tube], dtype=object)
    cls_ids = [x['cls_id'] for x in qft.qf_tube if x is not None]
    cls_id = max(set(cls_ids), key = cls_ids.count)
    return qf_feats_tube, cls_id

def get_tid_pairs(vid):
    # load qf tubes
    qf_path = osp.join(result_root, vid, "query_feats.pickle")
    qf_tubes = load_pickle(qf_path)
    # filter tubes with very few tracklets
    qf_tubes_filtered = list(filter(lambda x: valid_tube(x), qf_tubes))
    # get data and pair tids
    tid_pairs = []
    pair_tube_data = []
    tids_keep = []
    print("    Start pairing...", flush=True)
    for qft_s in tqdm(qf_tubes_filtered):
        for qft_o in qf_tubes_filtered:
            if qft_s.track_id == qft_o.track_id:
                continue
            if can_pair(qft_s, qft_o):
                qf_data_s, cls_id_s = generate_tube_data(qft_s)
                qf_data_o, cls_id_o = generate_tube_data(qft_o)
                tid_pairs.append([(qft_s.track_id, cls_id_s), (qft_o.track_id, cls_id_o)])
                pair_tube_data.append([qf_data_s, qf_data_o])
                tids_keep.append(qft_s.track_id)
                tids_keep.append(qft_o.track_id)
    tids_keep = sorted(list(set(tids_keep)))
    return tid_pairs, pair_tube_data, tids_keep

############################################################### mask ###############################################################
# load pred mask (copy from datasets/qf_tube_pair - postprocess)
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

def get_stacked_mask_vid(vid, tids_keep):
    # load pred masks
    mask_path = osp.join(result_root, vid, "quantitive/masks.txt")
    # group by tid
    pred_masks_grp_by_tid = load_pred_mask_tubes(mask_path)
    # get stacked mask for each tid in tids_keep
    pred_mask_vid = {}
    vid_len = gt[vid]['meta']['no_frames']
    h, w = gt[vid]['meta']['height'], gt[vid]['meta']['width']
    print("    Stacking masks...", flush=True)
    for tid in tqdm(tids_keep): # all tids to keep inside a singel video
        mask_list = []
        pred_masks = pred_masks_grp_by_tid[str(tid)]
        for frame_id in range(vid_len):
            mask_dict = next((item for item in pred_masks if item['fid'] == str(frame_id + 1)), None)
            if mask_dict is not None:
                mask_list.append(mask_dict['mask'])
            else:
                mask_list.append(np.zeros((h, w)))
        pred_mask_vid[tid] = np.stack(mask_list)
    return pred_mask_vid

####################################################################################################################################################

if __name__ == "__main__":
    # all data
    # all_video_data = {}
    for vid in vids:
        print("processing val video {}".format(vid), flush=True)
        tid_pairs, pair_tube_data, tids_keep = get_tid_pairs(vid)
        pred_mask_vid = get_stacked_mask_vid(vid, tids_keep)
        vid_dict = dict(tid_pairs=tid_pairs,
                        pair_tube_data=pair_tube_data,
                        tids_keep=tids_keep,
                        pred_mask_vid=pred_mask_vid)
        with open(f"/mnt/lustre/wxpeng/OpenPVSG/data/pvsg_v1_clean_qf_pair/full_val/ckpt8/{vid}.pickle", "wb") as f:
            pickle.dump(vid_dict, f)
    # # save
    # with open("/mnt/lustre/wxpeng/OpenPVSG/data/pvsg_v1_clean_qf_pair/val/ckpt7/all_video_data1.pickle", "wb") as f:
    #     pickle.dump(all_video_data, f)
