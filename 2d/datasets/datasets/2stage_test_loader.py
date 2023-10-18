import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import pickle
import logging
import sys
import math
from collections import defaultdict
from itertools import chain
from pathlib import Path
import json
from itertools import chain 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder

from sklearn.metrics import f1_score as f1_score

from typing import Dict

PREDICATES = ['behind', 'beside', 'biting', 'blowing', 'carrying', 'chasing', 
              'cutting', 'eating', 'grabbing', 'holding', 'hugging', 'in', 
              'in front of', 'kissing', 'licking', 'lighting', 'looking at', 
              'lying on', 'next to', 'on', 'over', 'petting', 'picking', 'playing with', 
              'pointing to', 'pulling', 'pushing', 'sitting on', 'sniffing', 
              'standing on', 'taking', 'talking to', 'throwing', 'touching', 'walking on', 'wearing']
THING_CLASSES = ['adult', 'baby', 'bag', 'ball', 'basket', 'bed', 'bike', 'book', 'bottle', 'bowl', 'box',
                 'cabinet', 'cake', 'camera', 'candle', 'car', 'cat', 'chair', 'child', 'cup', 'curtain', 'dog', 
                 'door', 'flower', 'fork', 'fridge', 'glass', 'hat', 'knife', 'light', 'lighter', 'mat', 
                 'paper', 'plant', 'plate', 'rock', 'shelf', 'shoe', 'sofa', 'table', 'towel', 'toy', 'tv', 'window']
STUFF_CLASSES = ['ceiling', 'fence', 'floor', 'grass', 'ground', 'pavement', 'road', 'sand', 'sky', 'tree', 'wall', 'water']

class SeqObj:
    """
    This is a seq object class for querying the image for constructing sequence.
    DIVISOR : This divisor is orthogonal with panoptic class-instance divisor (should be large enough).
    """
    DIVISOR = 1000000

    def __init__(self, the_dict: Dict):
        self.dict = the_dict
        assert 'seq_id' in self.dict and 'img_id' in self.dict

    def __hash__(self):
        return self.dict['seq_id'] * self.DIVISOR + self.dict['img_id']

    def __eq__(self, other):
        return self.dict['seq_id'] == other.dict['seq_id'] and self.dict['img_id'] == other.dict['img_id']

    def __getitem__(self, attr):
        return self.dict[attr]

class LoaderTestDatasetSingelVideo:
    def __init__(self, 
                 data_root="/mnt/lustre/wxpeng/OpenPVSG/data/pvsg_v1_clean_qf_pair/full_val/ckpt8",  # full val
                 vid="0005_2505076295",
                 mode="random",
                 **kwargs):
        self.mode = mode
        self.kwargs = kwargs
        self.data_path = Path(data_root) / f"{vid}.pickle"
        assert os.path.exists(self.data_path)
        with open(self.data_path, "rb") as f:
            data_dict = pickle.load(f)
        self.data = data_dict['pair_tube_data']
        self.tid_pairs = data_dict['tid_pairs']
        self.pred_masks = data_dict['pred_mask_vid']
        # one video have k tube pairs, still need seq_id
        self.ref_data_list = []
        self.seq2tid = {}
        for seq_id, (tid_pair, pair_tube) in enumerate(zip(self.tid_pairs, self.data)):
            self.seq2tid[seq_id] = tid_pair
            for frame_id, (frame_data_s, frame_data_o) in enumerate(zip(pair_tube[0], pair_tube[1])):
                self.ref_data_list.append(SeqObj({'seq_id': seq_id,
                                                  'img_id': frame_id,
                                                  'tid_pair': tid_pair,
                                                  'qf_s': frame_data_s,
                                                  'qf_o': frame_data_o,
                                                  'both_has_qf': frame_data_s is not None and frame_data_o is not None}))
        self.ref_data = {hash(x): x for x in self.ref_data_list}
        self.can_predict_ids = [hash(x) for x in self.ref_data_list if x.dict['both_has_qf']]

        
    def __getitem__(self, idx): # return pre_idx to remember which frame from which pair_tube is predicted!!
        pred_idx = self.can_predict_ids[idx]
        data_dict = self.ref_data[pred_idx].dict
        if self.mode == "random":
            qf_s, qf_o = data_dict['qf_s'], data_dict['qf_o']
            assert len(qf_s) == 256 and len(qf_o) == 256, "query feature dimension error!"
            data_concat = np.concatenate((qf_s, qf_s), axis=0)
            return pred_idx, data_concat
            
        if self.mode == "window": 
            ref_ids = self.kwargs['ref_ids'] # eg. [-2,-1,0,1,2]
            
            if self.kwargs['window_mode'] == "uniform":
                s_data_list = []
                o_data_list = []
                for ref_id in ref_ids:
                    ref_data_id = pred_idx + ref_id
                    if ref_data_id in self.ref_data and self.ref_data[ref_data_id].dict['both_has_qf']:
                        ref_data_dict = self.ref_data[ref_data_id].dict
                        s_data_list.append(ref_data_dict['qf_s'])
                        o_data_list.append(ref_data_dict['qf_o'])
                s_data_avg = np.average(s_data_list, axis=0)
                o_data_avg = np.average(o_data_list, axis=0)
                data_concat = np.concatenate((s_data_avg, o_data_avg), axis=0)
                return pred_idx, data_concat
                    
            if self.kwargs['window_mode'] == "weighted":
                s_data_list = []
                o_data_list = []
                weights = []
                for ref_id in ref_ids:
                    ref_data_id = pred_idx + ref_id
                    if ref_data_id in self.ref_data and self.ref_data[ref_data_id].dict['both_has_qf']:
                        ref_data_dict = self.ref_data[ref_data_id].dict
                        s_data_list.append(ref_data_dict['qf_s'])
                        o_data_list.append(ref_data_dict['qf_o'])
                        weights.append(2**(-abs(ref_id)))
                s_data_avg = np.average(s_data_list, axis=0, weights=weights)
                o_data_avg = np.average(o_data_list, axis=0, weights=weights)
                data_concat = np.concatenate((s_data_avg, o_data_avg), axis=0)
                return pred_idx, data_concat
        
        if self.mode == "transformer":
            ref_ids = self.kwargs['ref_ids'] # eg. [-2,-1,0,1,2]
            data_window = []
            output_pos = -1
            count = 0
            for ref_id in ref_ids: 
                if ref_id == 0:
                    output_pos = count  # remember the position of "0" data, the one we need to get output from transformer
                ref_data_id = pred_idx + ref_id
                if ref_data_id in self.ref_data and self.ref_data[ref_data_id].dict['both_has_qf']:
                    ref_data_dict = self.ref_data[ref_data_id].dict
                    data_concat = np.concatenate((ref_data_dict['qf_s'], ref_data_dict['qf_o']), axis=0)
                    data_window.append(data_concat)
                    count += 1
            data_window_arr = np.stack(data_window, axis=0) # [5,512]
            assert output_pos != -1, "0 must be in your ref ids!"
            return pred_idx, data_window_arr, output_pos
             
    
    def __len__(self):
        return len(self.can_predict_ids)

        
        

#################################### model ############################
device = 'cuda'
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerTemporalFC(nn.Module):
    def __init__(self, 
                 d_model=512,
                 n_head=8,
                 num_encoder_layers=4,
                 num_predicates=36
                 ):
        super(TransformerTemporalFC, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=0.1)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers)
        self.pe = PositionalEncoding(d_model=d_model)
        self.fc = nn.Linear(in_features=d_model,out_features=num_predicates)
        self.fc.apply(xavier)
        
    def forward(self, X, output_pos): # X already transpose -- [seq_len, batch_size, dim]
        X = self.pe(X)
        out_tfm = self.encoder(X)
        out_tfm = torch.transpose(out_tfm, 0, 1) # [batch_size, seq_len, dim]
        out0 = out_tfm[torch.arange(out_tfm.size(0)), output_pos]
        out = self.fc(out0)
        return out

############################ test ############################
checkpoint_root = "/mnt/lustre/wxpeng/OpenPVSG/2stage/full" # full!!!!!

def test_video(vid, mode="random", **kwargs):
    dataset = LoaderTestDatasetSingelVideo(vid=vid, mode=mode, **kwargs)
    if mode == "transformer":
        model = TransformerTemporalFC() # all default
        results = test_tfm(dataset=dataset, model=model, mode=mode)
    else:
        model = nn.Linear(in_features=512,out_features=36)
        if mode == "random":
            results = test(dataset=dataset, model=model, mode_dir_name="random")
        else: # window
            if kwargs['window_mode'] == "uniform":
                results = test(dataset=dataset, model=model, mode_dir_name="uniform_window")
            else:
                results = test(dataset=dataset, model=model, mode_dir_name="weighted_window")

    return dataset, results

# for mode exclude transformer
def test(dataset, model, mode_dir_name = "random"):
    checkpoint_path = osp.join(checkpoint_root, mode_dir_name, "epoch50.pth")
    assert os.path.exists(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device=device, dtype=torch.float64)
    model.eval()
    results = {}
    with torch.no_grad():
        for pred_id, X in dataset:
            X = X.astype(np.float64)
            X = torch.from_numpy(X).to(device=device, dtype=torch.float64)
            X = X.unsqueeze(0) # add batch dimension
            out = model(X)  
            scores = F.softmax(out, dim=1)
            pred = (scores > 0.5).long().cpu().numpy()
            # only save pred with 1
            if pred.sum() > 0:
                results[pred_id] = np.squeeze(pred)
    return results

def test_tfm(dataset, model, mode = "transformer"):
    checkpoint_path = osp.join(checkpoint_root, mode, "epoch50.pth")
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device=device, dtype=torch.float64)
    model.eval()
    results = {}
    with torch.no_grad():
        for pred_id, X, output_pos in dataset:
            X = X.astype(np.float64)
            X = torch.from_numpy(X).to(device=device, dtype=torch.float64)
            X = X.unsqueeze(1) # add batch dimension in the middle !!!!! seq, batch, dim [5,1,512]
            output_pos = torch.Tensor([output_pos]).long().to(device=device)
            
            out = model(X, output_pos)  
            
            scores = F.softmax(out, dim=1)
            pred = (scores > 0.5).long().cpu().numpy()
            # only save pred with 1
            if pred.sum() > 0:
                results[pred_id] = np.squeeze(pred)
    return results

def get_all_pair_results(dataset, results):
    ref_data = dataset.ref_data
    seq2tid = dataset.seq2tid

    pair_results = []
    for seq_id, tid_pairs in seq2tid.items():
        ref_data_this_seq = {k: v for k, v in ref_data.items() if k // 1000000 == seq_id}    # this pair
        pred_this_seq = []
        for hash_id in ref_data_this_seq.keys():
            if hash_id in results:
                pred_this_seq.append(array2label_id(results[hash_id]))
            else:
                pred_this_seq.append([])
        pair_results.append(dict(tid_pair=tid_pairs,
                                 pred=pred_this_seq))
    return pair_results, dataset.pred_masks

def array2label_id(pred_arr):
    return list(np.where(pred_arr == 1)[0])

def get_clips_for_one_predicate_id(p_id, pred_list): # pred_list - pred result for entire video of one pair
    pred_indicator = [1 if p_id in x else 0 for x in pred_list]
    appear_frame_ids = list(np.where(np.array(pred_indicator)==1)[0])
    durs = []
    cur_list = [appear_frame_ids[0]]
    for i in range(1, len(appear_frame_ids)):
        if appear_frame_ids[i] - 1 == appear_frame_ids[i-1]:
            cur_list.append(appear_frame_ids[i])
        else:
            durs.append(cur_list)
            cur_list = [appear_frame_ids[i]]
    durs.append(cur_list)

    clips = []
    for dur_list in durs:
        clips.append([dur_list[0], dur_list[-1]])
    return clips

def format_pred_results_per_pair(pair_results):
    all_pair_clips = []
    for pair_result in pair_results:
        tid_pair = pair_result['tid_pair']
        pred_list = pair_result['pred']
        appear_predicates = list(set(chain(*pred_list)))
        p_clips = {}
        for p in appear_predicates:
            clips = get_clips_for_one_predicate_id(p, pred_list)
            all_pair_clips.append(dict(triplet=(tid_pair[0], tid_pair[1], p),
                                       durs=clips))
    return all_pair_clips



if __name__ == "__main__":
    with open("/mnt/lustre/wxpeng/OpenPVSG/data/pvsg_v1_clean/val/val_video_names.json", "r") as f:
        vids = json.load(f)
    assert len(vids) == 18

    
    
    # random ##############################################################################################################################
    # for vid in tqdm(vids):
    #     dataset, results = test_video(vid=vid, mode="random")
    #     pair_results, pred_masks = get_all_pair_results(dataset, results)
    #     all_pair_clips = format_pred_results_per_pair(pair_results)
    #     vid_result = dict(all_pair_clips=all_pair_clips,pred_masks=pred_masks)
    #     with open(f"/mnt/lustre/wxpeng/OpenPVSG/data/metric_input_ckpt50/random/{vid}.pickle", "wb") as f:
    #         pickle.dump(vid_result, f)

    # uniform window ################################################################################################################
    # for vid in tqdm(vids):
    #     dataset, results = test_video(vid=vid, mode="window", ref_ids = [-2,-1,0,1,2], window_mode="uniform")
    #     pair_results, pred_masks = get_all_pair_results(dataset, results)
    #     all_pair_clips = format_pred_results_per_pair(pair_results)
    #     vid_result = dict(all_pair_clips=all_pair_clips,pred_masks=pred_masks)
    #     with open(f"/mnt/lustre/wxpeng/OpenPVSG/data/metric_input_ckpt50/uniform_window/{vid}.pickle", "wb") as f:
    #         pickle.dump(vid_result, f)


    # weigted window ################################################################################################################

    # for vid in tqdm(vids):
    #     dataset, results = test_video(vid=vid, mode="window", ref_ids = [-2,-1,0,1,2], window_mode="weighted")
    #     pair_results, pred_masks = get_all_pair_results(dataset, results)
    #     all_pair_clips = format_pred_results_per_pair(pair_results)
    #     vid_result = dict(all_pair_clips=all_pair_clips,pred_masks=pred_masks)
    #     with open(f"/mnt/lustre/wxpeng/OpenPVSG/data/metric_input_ckpt50/weighted_window/{vid}.pickle", "wb") as f:
    #         pickle.dump(vid_result, f)

    # transformer ################################################################################################################
    for vid in tqdm(vids):
        dataset, results = test_video(vid=vid, mode="transformer", ref_ids = [-2,-1,0,1,2])
        pair_results, pred_masks = get_all_pair_results(dataset, results)
        all_pair_clips = format_pred_results_per_pair(pair_results)
        vid_result = dict(all_pair_clips=all_pair_clips,pred_masks=pred_masks)
        with open(f"/mnt/lustre/wxpeng/OpenPVSG/data/metric_input_ckpt50/transformer/{vid}.pickle", "wb") as f:
            pickle.dump(vid_result, f)
