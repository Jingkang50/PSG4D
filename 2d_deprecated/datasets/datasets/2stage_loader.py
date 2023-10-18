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

# dummy dataset
def load_data_from_file(data_path, label_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(label_path, "rb") as f:
        label = pickle.load(f)
    return data, label

class LoaderDataset(Dataset):
    def __init__(self, 
                 data_root="/mnt/lustre/wxpeng/OpenPVSG/data/pvsg_v1_clean_qf_pair/full_train/ckpt8",
                 mode="random",
                 split="train",
                 data_filename="data.pickle",
                 label_filename="label.pickle",
                 **kwargs): # kwargs - window_mode="uniform" or "weighted"; ref_ids=[-2,-1,0,1,2]... if window mode
        self.mode = mode
        self.split = split
        self.kwargs = kwargs
        # load data and label from file
        # data_path = os.path.join(data_root, f'{split}_{data_filename}')
        # label_path = os.path.join(data_root, f'{split}_{label_filename}')
        data_path = os.path.join(data_root, data_filename)
        label_path = os.path.join(data_root, label_filename)
        self.data, self.label = load_data_from_file(data_path, label_path)
        # get reference data and hash index
        ref_data_list = []
        for seq_id, (tube_data, tube_label) in enumerate(zip(self.data, self.label)):
            # here seq_id means id of pair tube (not video, one video may have multiple pair tubes)
            tube_len = len(tube_label)
            for frame_id, (frame_data_s, frame_data_o, frame_label) in enumerate(zip(tube_data[0], tube_data[1], tube_label)):  ## bug!!!! [1]!!!!
                ref_data_list.append(SeqObj({'seq_id': seq_id,
                                             'img_id': frame_id,
                                             'qf_s': frame_data_s,
                                             'qf_o': frame_data_o,
                                             'label': frame_label,
                                             'both_has_qf': frame_data_s is not None and frame_data_o is not None,
                                             'has_label': frame_label is not None}))  
        self.ref_data = {hash(x): x for x in ref_data_list}
        self.train_data_ids = [hash(x) for x in ref_data_list if x.dict['has_label']]
        
    
    def __getitem__(self, idx):
        train_idx = self.train_data_ids[idx]
        data_dict = self.ref_data[train_idx].dict
        if self.mode == "random":
            qf_s, qf_o, label = data_dict['qf_s'], data_dict['qf_o'], data_dict['label']
            assert len(qf_s) == 256 and len(qf_o) == 256, "query feature dimension error!"
            data_concat = np.concatenate((qf_s, qf_s), axis=0)
            return data_concat, label
            
        if self.mode == "window": 
            ref_ids = self.kwargs['ref_ids'] # eg. [-2,-1,0,1,2]
            
            if self.kwargs['window_mode'] == "uniform":
                s_data_list = []
                o_data_list = []
                for ref_id in ref_ids:
                    ref_data_id = train_idx + ref_id
                    if ref_data_id in self.ref_data and self.ref_data[ref_data_id].dict['both_has_qf']:
                        ref_data_dict = self.ref_data[ref_data_id].dict
                        s_data_list.append(ref_data_dict['qf_s'])
                        o_data_list.append(ref_data_dict['qf_o'])
                s_data_avg = np.average(s_data_list, axis=0)
                o_data_avg = np.average(o_data_list, axis=0)
                data_concat = np.concatenate((s_data_avg, o_data_avg), axis=0)
                return data_concat, data_dict['label']
                    
            if self.kwargs['window_mode'] == "weighted":
                s_data_list = []
                o_data_list = []
                weights = []
                for ref_id in ref_ids:
                    ref_data_id = train_idx + ref_id
                    if ref_data_id in self.ref_data and self.ref_data[ref_data_id].dict['both_has_qf']:
                        ref_data_dict = self.ref_data[ref_data_id].dict
                        s_data_list.append(ref_data_dict['qf_s'])
                        o_data_list.append(ref_data_dict['qf_o'])
                        weights.append(2**(-abs(ref_id)))
                s_data_avg = np.average(s_data_list, axis=0, weights=weights)
                o_data_avg = np.average(o_data_list, axis=0, weights=weights)
                data_concat = np.concatenate((s_data_avg, o_data_avg), axis=0)
                return data_concat, data_dict['label']
        
        if self.mode == "transformer":
            ref_ids = self.kwargs['ref_ids'] # eg. [-2,-1,0,1,2]
            data_window = []
            output_pos = -1
            count = 0
            for ref_id in ref_ids: 
                if ref_id == 0:
                    output_pos = count  # remember the position of "0" data, the one we need to get output from transformer
                ref_data_id = train_idx + ref_id
                if ref_data_id in self.ref_data and self.ref_data[ref_data_id].dict['both_has_qf']:
                    ref_data_dict = self.ref_data[ref_data_id].dict
                    data_concat = np.concatenate((ref_data_dict['qf_s'], ref_data_dict['qf_o']), axis=0)
                    data_window.append(data_concat)
                    count += 1
            data_window_arr = np.stack(data_window, axis=0) # [5,512]
            assert output_pos != -1, "0 must be in your ref ids!"
            return data_window_arr, data_dict['label'], output_pos
                
            
            
            
            
            
#         tube_version
#         if self.mode == "transformer": # return a tube of data
#             pair_tube = self.data[idx]
#             label_tube = self.label[idx]
#             tube_concat = np.array([np.concatenate((qf_s, qf_s), axis=0) \
#                            if qf_s is not None and qf_o is not None else np.zeros((512)) for qf_s, qf_o in zip(pair_tube[0], pair_tube[1])])
#             # train_ids = [i for i, _label in enumerate(label_tube) if _label is not None]
#             dummy_label = np.zeros(32).astype(np.int64)
#             label_tube_none2zeros = np.array([l if l is not None else dummy_label for l in label_tube])
#             return tube_concat, label_tube_none2zeros
        
             
    
    def __len__(self):
        return len(self.train_data_ids)
#         # tube version of transformer
#         if self.mode == "transformer":
#             return len(self.data)
#         else:
#             return len(self.train_data_ids)

    


########################################### functions for simple trainig ##################################################
device = 'cuda'
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def load_data(data_root="/mnt/lustre/wxpeng/OpenPVSG/data/pvsg_v1_clean_qf_pair/train/ckpt7",
              mode="random",
              split="train",
              data_filename="data.pickle",
              label_filename="label.pickle",
              batch_size=256,
              num_workers=4,
              **kwargs) -> DataLoader:
    dataset = LoaderDataset(data_root=data_root,
                            mode=mode,
                            split=split,
                            data_filename=data_filename,
                            label_filename=label_filename,
                            **kwargs)
    return DataLoader(dataset=dataset, 
                      batch_size=batch_size, 
                      shuffle=(split=="train"), 
                      num_workers=num_workers)

def train(model, train_dataloader, val_dataloader, optimizer, num_epochs, print_every, save_dir):
    model = model.to(device=device)
    model.train()
    best_f1_score = -1
    for epoch in range(num_epochs):
        save_best = False
        for i, (X, y) in enumerate(train_dataloader):
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            out = model(X)
            scores = F.softmax(out, dim=1)
            #loss = F.cross_entropy(scores, y)
            loss = F.binary_cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % print_every == 0:
                logging.info(f"epoch {epoch}, iteration {i}, loss = {loss}")
                logging.info("  train f1 score:")
                train_f1_score = compute_f1_score(model, train_dataloader)
                # logging.info("  val f1 score:")
                # compute_f1_score(model, val_dataloader)
                if train_f1_score > best_f1_score:
                    best_f1_score = train_f1_score
                    save_best = True
        torch.save(model.state_dict(), osp.join(save_dir, f"epoch{epoch}.pth"))
        if save_best:
            torch.save(model.state_dict(), osp.join(save_dir, f"best.pth"))
                
def compute_f1_score(model, dataloader):
    model.eval()
    with torch.no_grad():
        f1_s = 0.0
        count = 0
        for X, y in dataloader:
            X = X.to(device=device, dtype=torch.float32)
            out = model(X)
            scores = F.softmax(out, dim=1)
            preds = (scores > 0.5).long().cpu().numpy()
            
            for _pred, _gt in zip(preds, y):
                f1_s += f1_score(_gt, _pred)
                count += 1
        f1_s = f1_s / count
        logging.info(f"    f1 score = {f1_s}")
    return f1_s

def log_result(filename):
    old_stdout = sys.stdout
    log_file = open(filename,"w")
    sys.stdout = log_file
    print(f"this will be written to {filename}")
    sys.stdout = old_stdout
    log_file.close()
    

## train for transformer *************************
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

def train_tfm(model, train_dataloader, tfm_encoder, pe, val_dataloader, optimizer, num_epochs, print_every):
    ##### trick here !##############################
    train_dataloader = list(train_dataloader)
    val_dataloader = list(val_dataloader)
    ##################################################

    tfm_encoder = tfm_encoder.to(device=device, dtype=torch.float64)
    pe = pe.to(device=device, dtype=torch.float64)
    model = model.to(device=device, dtype=torch.float64)
    model.train()
    tfm_encoder.train()
    for epoch in range(num_epochs):
        for i, (X, y, output_pos) in enumerate(train_dataloader):
            X = X.transpose(0,1) # [seq_len, batch_size, dim]
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.double)
            output_pos = output_pos.to(device=device)
            X = pe(X) # add positional encoding
            out_tfm = tfm_encoder(X) # [seq_len, batch_size, dim]
            # import pdb; pdb.set_trace()
            ################################################################################
            out_tfm = torch.transpose(out_tfm, 0, 1) # [batch_size, seq_len, dim]
            out0 = out_tfm[torch.arange(out_tfm.size(0)), output_pos]
            ################################################################################
            # out0 = out_tfm[output_pos.item()]
            out = model(out0)
            scores = F.softmax(out, dim=1)
            #loss = F.cross_entropy(scores, y)
            loss = F.binary_cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % print_every == 0:
                logging.info(f"epoch {epoch}, iteration {i}, loss = {loss}")
                logging.info("  train f1 score:")
                compute_f1_score_tfm(model, encoder=tfm_encoder, pe=pe, dataloader=train_dataloader)
                logging.info("  val f1 score:")
                compute_f1_score_tfm(model, encoder=tfm_encoder, pe=pe, dataloader=val_dataloader)
                
def compute_f1_score_tfm(model, encoder, pe, dataloader):
    model.eval()
    with torch.no_grad():
        f1_s = 0.0
        count = 0
        for X, y, output_pos in dataloader:
            X = X.transpose(0,1)
            X = X.to(device=device, dtype=torch.float32)
            output_pos = output_pos.to(device=device)
            X = pe(X)
            out_tfm = encoder(X)
            # out0 = out_tfm[output_pos.item()]
            out_tfm = torch.transpose(out_tfm, 0, 1) # [batch_size, seq_len, dim]
            out0 = out_tfm[torch.arange(out_tfm.size(0)), output_pos]
            out = model(out0)
            scores = F.softmax(out, dim=1)
            preds = (scores > 0.5).long().cpu().numpy()
            
            for _pred, _gt in zip(preds, y):
                f1_s += f1_score(_gt, _pred)
                count += 1
        f1_s = f1_s / count
        logging.info(f"    f1 score = {f1_s}")
 # ********************************************

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* train merged transoformer + fc model-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
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

def compute_f1_score_tfm_fc(model, dataloader):
    model.eval()
    with torch.no_grad():
        f1_s = 0.0
        count = 0
        for X, y, output_pos in dataloader:
            X = X.transpose(0,1)
            X = X.to(device=device, dtype=torch.float32)
            output_pos = output_pos.to(device=device)
            
            out = model(X, output_pos)
            
            scores = F.softmax(out, dim=1)
            preds = (scores > 0.5).long().cpu().numpy()
            
            for _pred, _gt in zip(preds, y):
                f1_s += f1_score(_gt, _pred)
                count += 1
        f1_s = f1_s / count
        logging.info(f"    f1 score = {f1_s}")
        return f1_s

def train_tfm_fc(model, train_dataloader, optimizer, num_epochs, print_every, save_dir):
    train_dataloader = list(train_dataloader)
    
    model = model.to(device=device, dtype=torch.float64) # merged model
    model.train()
    best_f1_score = -1
    for epoch in range(num_epochs):
        save_best = False
        for i, (X, y, output_pos) in enumerate(train_dataloader):
            X = X.transpose(0,1) # [seq_len, batch_size, dim]
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.double)
            output_pos = output_pos.to(device=device)
            
            out = model(X, output_pos)
            
            scores = F.softmax(out, dim=1)
            #loss = F.cross_entropy(scores, y)
            loss = F.binary_cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % print_every == 0:
                logging.info(f"epoch {epoch}, iteration {i}, loss = {loss}")
                logging.info("  train f1 score:")
                train_f1_score = compute_f1_score_tfm_fc(model=model, dataloader=train_dataloader)
                if train_f1_score > best_f1_score:
                    best_f1_score = train_f1_score
                    save_best = True
                    
        torch.save(model.state_dict(), osp.join(save_dir, f"epoch{epoch}.pth"))
        if save_best:
            torch.save(model.state_dict(), osp.join(save_dir, f"best.pth"))
    


# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*





#################################################################################################################################

if __name__ == "__main__":
    # train window mode - weighted avg #########################################################################
    # train_dataloader = load_data(mode="window", ref_ids=[-2,-1,0,1,2], window_mode="weighted")
    # # val_dataloader = load_data(mode="window", split="val", ref_ids=[-2,-1,0,1,2], window_mode="weighted")
    # # fc = FC_Layer(in_features=512, out_features=32)
    # model = nn.Linear(in_features=512,out_features=36)
    # model.apply(xavier)
    # lr = 0.001
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01, nesterov=True)
    # logging.basicConfig(level=logging.DEBUG, filename="/mnt/lustre/wxpeng/OpenPVSG/2stage/full/weighted_window/loss&f1_scores.log", filemode="a+",
    #                     format="%(asctime)-15s %(levelname)-8s %(message)s")
    # train(model=model, train_dataloader=train_dataloader, val_dataloader=None, optimizer=optimizer, num_epochs=150, print_every=100, save_dir="/mnt/lustre/wxpeng/OpenPVSG/2stage/full/weighted_window")

    # train window mode - uniform #########################################################################
    # train_dataloader = load_data(mode="window", ref_ids=[-2,-1,0,1,2], window_mode="uniform")
    # # val_dataloader = load_data(mode="window", split="val", ref_ids=[-2,-1,0,1,2], window_mode="uniform")
    # # fc = FC_Layer(in_features=512, out_features=32)
    # model = nn.Linear(in_features=512,out_features=36)
    # model.apply(xavier)
    # lr = 0.001
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01, nesterov=True)
    # logging.basicConfig(level=logging.DEBUG, filename="/mnt/lustre/wxpeng/OpenPVSG/2stage/full/uniform_window/loss&f1_scores.log", filemode="a+",
    #                      format="%(asctime)-15s %(levelname)-8s %(message)s")
    # train(model=model, train_dataloader=train_dataloader, val_dataloader=None, optimizer=optimizer, num_epochs=150, print_every=100, save_dir="/mnt/lustre/wxpeng/OpenPVSG/2stage/full/uniform_window")

    # # train random #########################################################################
    # train_dataloader = load_data(mode="random")
    # # val_dataloader = load_data(mode="random", split="val")
    # # fc = FC_Layer(in_features=512, out_features=32)
    # model = nn.Linear(in_features=512,out_features=36)
    # model.apply(xavier)
    # lr = 0.001
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01, nesterov=True)
    # logging.basicConfig(level=logging.DEBUG, filename="/mnt/lustre/wxpeng/OpenPVSG/2stage/full/random/loss&f1_scores.log", filemode="a+",
    #                      format="%(asctime)-15s %(levelname)-8s %(message)s")
    # train(model=model, train_dataloader=train_dataloader, val_dataloader=None, optimizer=optimizer, num_epochs=150, print_every=100, save_dir="/mnt/lustre/wxpeng/OpenPVSG/2stage/full/random")

    ###### train transformer
    # device = 'cuda'rando
    # train_dataloader = load_data(mode="transformer", ref_ids=[-2,-1,0,1,2], batch_size=1, num_workers=1)
    # val_dataloader = load_data(mode="transformer",split='val', ref_ids=[-2,-1,0,1,2], batch_size=1, num_workers=1)
    # # fc = FC_Layer(in_features=512, out_features=32)
    # model = nn.Linear(in_features=512,out_features=32)
    # model.apply(xavier)

    # encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1)
    # encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)
    # pe = PositionalEncoding(d_model=512)

    # lr = 0.001
    # optimizer = optim.SGD([{'params': model.parameters()},{'params':encoder.parameters()}], lr=lr, momentum=0.9, weight_decay=0.01, nesterov=True)
    # logging.basicConfig(level=logging.DEBUG, filename="/mnt/lustre/wxpeng/OpenPVSG/r_transformer_worker1_printevery10000.log", filemode="a+",
    #                      format="%(asctime)-15s %(levelname)-8s %(message)s")
    # train_tfm(model=model, train_dataloader=train_dataloader, tfm_encoder=encoder, pe=pe, val_dataloader=val_dataloader, optimizer=optimizer, num_epochs=150, print_every=10000)

    ###### train transformer - chained dataloader
    # chain train dataloader
    device = 'cuda'
    train_dataset = LoaderDataset(mode="transformer", ref_ids=[-2,-1,0,1,2])
    data_batch = defaultdict(list)
    for data in train_dataset:
        data_batch[data[0].shape[0]].append(data)
    dataloader1 = DataLoader(dataset=data_batch[1], batch_size=256, shuffle=True, num_workers=4)
    dataloader2 = DataLoader(dataset=data_batch[2], batch_size=256, shuffle=True, num_workers=4)
    dataloader3 = DataLoader(dataset=data_batch[3], batch_size=256, shuffle=True, num_workers=4)
    dataloader4 = DataLoader(dataset=data_batch[4], batch_size=256, shuffle=True, num_workers=4)
    dataloader5 = DataLoader(dataset=data_batch[5], batch_size=256, shuffle=True, num_workers=4)
    train_dataloader = chain(dataloader1, dataloader2, dataloader3, dataloader4, dataloader5)
    
    # fc = FC_Layer(in_features=512, out_features=32)
    model = TransformerTemporalFC(d_model=512, n_head=8, num_encoder_layers=4, num_predicates=36)

    lr = 0.001
    optimizer = optim.SGD([{'params': model.parameters()}], lr=lr, momentum=0.9, weight_decay=0.01, nesterov=True)
        
    logging.basicConfig(level=logging.DEBUG, filename="/mnt/lustre/wxpeng/OpenPVSG/2stage/full/transformer/loss&f1_scores.log", filemode="a+",
                         format="%(asctime)-15s %(levelname)-8s %(message)s")
    train_tfm_fc(model=model, train_dataloader=train_dataloader, optimizer=optimizer, num_epochs=150, print_every=100, save_dir="/mnt/lustre/wxpeng/OpenPVSG/2stage/full/transformer")
    print("training finished!")
