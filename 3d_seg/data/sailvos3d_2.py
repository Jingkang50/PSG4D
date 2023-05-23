'''
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by wuyizheng
'''

import os, sys, glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import DataLoader
import multiprocessing as mp
from PIL import Image
import random
import matplotlib.pyplot as plt

from typing import Dict, List, Sequence, Tuple, Union
import open3d as o3d

sys.path.append('../')

import time
from utils.config import cfg
from utils.log import logger
from utils.timer import Timer

import segmentator
import dknet_ops
import json

class GetSuperpoint(mp.Process):
    def __init__(self, path: str, scene: str, mdict: Dict):
        # must call this before anything else
        mp.Process.__init__(self)
        self.path = path
        self.scene = scene
        self.mdict = mdict

    def run(self):
        mesh_file = os.path.join(os.path.join(self.path, self.scene+"_vh_clean_2.ply"))
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
        faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
        superpoint = segmentator.segment_mesh(vertices, faces).numpy()
        self.mdict.update({self.scene: superpoint})      

class Dataset:
    def __init__(self, test=False):
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.filename_suffix = cfg.filename_suffix

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode
        self.prefetch_superpoints = cfg.prefetch_superpoints

        if test or cfg.test_epoch:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            self.test_batch_size = cfg.test_batch_size


    def trainLoader(self):
        self.images_train_path = []
        self.depth_train_path = []
        self.rage_matrices_train_path = []
        self.visible_train_path = []
        for line in open("/mnt/lustre/jkyang/PSG4D/dknet/dataset/sailvos3d/train_20.txt"):
            self.rage_matrices_train_path.append(line[:-1])
        ids = random.sample(self.rage_matrices_train_path, int(len(self.rage_matrices_train_path) / 1))
        for id in ids:
            prefix = id[:-24]
            id_num = id[-10:-3]
            self.images_train_path.append(prefix + "images/" + id_num + "bmp")
            self.depth_train_path.append(prefix + "depth/" + id_num + "npy")
            self.visible_train_path.append(prefix + "/visible/" + id_num + "npy")
        self.train_files = self.rage_matrices_train_path
        
        with open('/mnt/lustre/jkyang/PSG4D/liushuai_workdir/psg4d_val_v1.json', 'r') as fcc_file:
            self.labels_json_file = json.load(fcc_file)
        object_to_cls = {}
        for obj in self.labels_json_file['object']:
            object_to_cls[obj] = self.labels_json_file['object'].index(obj)
        self.labels_dict = {}
        for dict_tmp in self.labels_json_file['data']:
            object_dict_new = {}
            for objects_dict in dict_tmp['object']:
                object_dict_new[objects_dict['object_id']] = object_to_cls[objects_dict['category']]
            self.labels_dict[dict_tmp['video_id']] = object_dict_new

        logger.info('Training samples: {}'.format(len(self.train_files)))
        self.superpoints = {}
 
        if self.prefetch_superpoints:
            logger.info("begin prefetch superpoints...")
            path = os.path.join(self.data_root, self.dataset, 'train')
            with Timer("prefetch superpoints:"):
                workers = []
                mdict = mp.Manager().dict()
                # multi-processing generate superpoints
                for i in range(len(self.train_file_names)):
                    workers.append(GetSuperpoint(path, self.train_file_names[i].split('/')[-1][:12], mdict))
                for worker in workers:
                    worker.start()
                # wait for multi-processing
                while len(mdict) != len(self.tran_files):
                    time.sleep(0.1)
                self.superpoints.update(mdict)

        train_set = list(np.arange(len(self.rage_matrices_train_path)))
        if cfg.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        else:
            train_sampler = None

        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers,
                                            shuffle=(train_sampler is None), sampler=train_sampler, drop_last=True, pin_memory=True)

    def valLoader(self):
        self.images_val_path = []
        self.depth_val_path = []
        self.rage_matrices_val_path = []
        self.visible_val_path = []
        for line in open("/mnt/lustre/jkyang/PSG4D/dknet/dataset/sailvos3d/val_20.txt"):
            self.rage_matrices_val_path.append(line[:-1])
        ids = random.sample(self.rage_matrices_val_path, int(len(self.rage_matrices_val_path) / 1))
        for id in ids:
            prefix = id[:-24]
            id_num = id[-10:-3]
            self.images_val_path.append(prefix + "images/" + id_num + "bmp")
            self.depth_val_path.append(prefix + "depth/" + id_num + "npy")
            self.visible_val_path.append(prefix + "/visible/" + id_num + "npy")
        self.val_files = self.rage_matrices_val_path
        
        with open('/mnt/lustre/jkyang/PSG4D/liushuai_workdir/psg4d_val_v1.json', 'r') as fcc_file:
            self.labels_json_file = json.load(fcc_file)
        object_to_cls = {}
        for obj in self.labels_json_file['object']:
            object_to_cls[obj] = self.labels_json_file['object'].index(obj)
        self.labels_dict = {}
        for dict_tmp in self.labels_json_file['data']:
            object_dict_new = {}
            for objects_dict in dict_tmp['object']:
                object_dict_new[objects_dict['object_id']] = object_to_cls[objects_dict['category']]
            self.labels_dict[dict_tmp['video_id']] = object_dict_new

        logger.info('Validation samples: {}'.format(len(self.val_files)))
        self.superpoints = {}
 
        if self.prefetch_superpoints:
            logger.info("begin prefetch superpoints...")
            path = os.path.join(self.data_root, self.dataset, 'val')
            with Timer("prefetch superpoints:"):
                workers = []
                mdict = mp.Manager().dict()
                # multi-processing generate superpoints
                for i in range(len(self.val_file_names)):
                    workers.append(GetSuperpoint(path, self.val_file_names[i].split('/')[-1][:12], mdict))
                for worker in workers:
                    worker.start()
                # wait for multi-processing
                while len(mdict) != len(self.val_files):
                    time.sleep(0.1)
                self.superpoints.update(mdict)

        val_set = list(np.arange(len(self.rage_matrices_val_path)))
        if cfg.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        else:
            val_sampler = None

        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers,
                                            shuffle=False, sampler=val_sampler, drop_last=False, pin_memory=True)
                                            
    def testLoader(self):

        self.images_test_path = []
        self.depth_test_path = []
        self.rage_matrices_test_path = []
        self.visible_test_path = []
        for line in open("/mnt/lustre/jkyang/PSG4D/dknet/dataset/sailvos3d/val_20.txt"):
            self.rage_matrices_test_path.append(line[:-1])
        ids = random.sample(self.rage_matrices_test_path, int(len(self.rage_matrices_test_path) / 1))
        for id in ids:
            prefix = id[:-24]
            id_num = id[-10:-3]
            self.images_test_path.append(prefix + "images/" + id_num + "bmp")
            self.depth_test_path.append(prefix + "depth/" + id_num + "npy")
            self.visible_test_path.append(prefix + "/visible/" + id_num + "npy")
        self.test_files = self.rage_matrices_test_path
        
        with open('/mnt/lustre/jkyang/PSG4D/liushuai_workdir/psg4d_val_v1.json', 'r') as fcc_file:
            self.labels_json_file = json.load(fcc_file)
        object_to_cls = {}
        for obj in self.labels_json_file['object']:
            object_to_cls[obj] = self.labels_json_file['object'].index(obj)
        self.labels_dict = {}
        for dict_tmp in self.labels_json_file['data']:
            object_dict_new = {}
            for objects_dict in dict_tmp['object']:
                object_dict_new[objects_dict['object_id']] = object_to_cls[objects_dict['category']]
            self.labels_dict[dict_tmp['video_id']] = object_dict_new

        logger.info('Testing samples ({}): {}'.format(self.test_split, len(self.test_files)))
        self.superpoints = {}
 
        if self.prefetch_superpoints:
            logger.info("begin prefetch superpoints...")
            path = os.path.join(self.data_root, self.dataset, self.test_split)
            with Timer("prefetch superpoints:"):
                workers = []
                mdict = mp.Manager().dict()
                # multi-processing generate superpoints
                for i in range(len(self.test_file_names)):
                    workers.append(GetSuperpoint(path, self.test_file_names[i].split('/')[-1][:12], mdict))
                for worker in workers:
                    worker.start()
                # wait for multi-processing
                while len(mdict) != len(self.test_files):
                    time.sleep(0.1)
                self.superpoints.update(mdict)

        #test_set = list(np.arange(cfg.test_batch_size*len(self.test_files)))
        test_set = list(np.arange(len(self.rage_matrices_test_path)))
        self.test_data_loader = DataLoader(test_set, batch_size=cfg.test_batch_size, collate_fn=self.testMerge, num_workers=self.test_workers,
                                           shuffle=False, drop_last=False, pin_memory=True)

    def get_superpoint(self, scene: str, sub_dir: str):
        if scene in self.superpoints:
            return
        mesh_file = os.path.join(self.data_root, self.dataset, sub_dir, scene+"_vh_clean_2.ply")
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
        faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
        superpoint = segmentator.segment_mesh(vertices, faces).numpy()
        self.superpoints[scene] = superpoint

    #Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x + g(x) * mag


    def getInstanceInfo(self, xyz, instance_label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # (nInst), int
        #instance_center = np.zeros(xyz.shape[0], dtype=int)
        instance_center = []
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            center_i = np.argmin(np.abs(xyz_i - mean_xyz_i).sum(1))
            #print(instance_center.size, center_i)
            instance_center.append(inst_idx_i[0][center_i])

            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum, "instance_center": instance_center}


    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)


    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs


    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label


    def trainMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int
        instance_center = [] # (total_nInst, 3), float
        superpoint_list = []
        range_xyz = []
        superpoint_bias = 0

        batch_offsets = [0]

        total_inst_num = 0
        valid_data = True
        for i, idx in enumerate(id):
            # xyz_origin, rgb, label, instance_label = self.train_files[idx]
            xyz_origin, rgb, label, instance_label = self.read_sailvos3d(self.images_train_path, self.depth_train_path, self.rage_matrices_train_path, self.visible_train_path, self.labels_dict, idx)
            range_xyz.append(xyz_origin.min(0))
            range_xyz.append(xyz_origin.max(0))
            # scene = self.train_file_names[idx].split('/')[-1][:12]
            # if not self.prefetch_superpoints:
            #     self.get_superpoint(scene, 'train')
            # superpoint = self.superpoints[scene]
            superpoint = np.zeros_like(label)

            ### jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, False, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### elastic
            # xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            # xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

            ### offset
            # print(self.images_train_path[idx], 'xyz', xyz.shape)
            xyz -= xyz.min(0)

            ### crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            superpoint += superpoint_bias
            superpoint_bias += (superpoint.max() + 1)

            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]   # (nInst), list
            inst_center = inst_infos['instance_center']

            inst_center = [center + batch_offsets[-1] for center in inst_center]
            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))
            superpoint_list.append(torch.from_numpy(superpoint)) 

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)
            #print(instance_center)
            instance_center.extend(inst_center)
            #print(instance_center)

        # if valid_data == False:
        #     return None
        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        superpoint = torch.cat(superpoint_list, 0).long()               # long[N]
        feats = torch.cat(feats, 0).float()                                 # float (N, C)
        labels = torch.cat(labels, 0).long()                     # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()   # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        instance_center = torch.tensor(instance_center, dtype=torch.float32)
        instance_mask = torch.zeros((instance_pointnum.shape[0], len(labels)), dtype = torch.int) # int (total_nInst, N)
        for i in range(instance_mask.shape[0]): instance_mask[i] = (instance_labels == i).int()  
        #print(instance_center.shape, instance_center.sum(), instance_pointnum.shape[0])

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = dknet_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum, 'instance_mask': instance_mask,
                'instance_center': instance_center, 'id': [self.rage_matrices_train_path[idx] for idx in id], 'offsets': batch_offsets, 'spatial_shape': spatial_shape, "superpoint": superpoint,
                'range': range_xyz}


    def valMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int
        instance_center = [] # (total_nInst, 3), float
        superpoint_list = []
        superpoint_bias = 0

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):
            # xyz_origin, rgb, label, instance_label = self.val_files[idx]
            xyz_origin, rgb, label, instance_label = self.read_sailvos3d(self.images_val_path, self.depth_val_path, self.rage_matrices_val_path, self.visible_val_path, self.labels_dict, idx)


            # scene = self.val_file_names[idx].split('/')[-1][:12]
            # if not self.prefetch_superpoints:
            #     self.get_superpoint(scene, 'val')
            # superpoint = self.superpoints[scene]
            superpoint = np.zeros_like(label)

            ### jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, True, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### elastic
            xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

            ### offset
            xyz -= xyz.min(0)

            ### crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            superpoint += superpoint_bias
            superpoint_bias += (superpoint.max() + 1)

            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]   # (nInst), list
            inst_center = inst_infos['instance_center']

            inst_center = [center + batch_offsets[-1] for center in inst_center]
            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))
            superpoint_list.append(torch.from_numpy(superpoint)) 

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)
            #print(instance_center)
            instance_center.extend(inst_center)
            #print(instance_center)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        superpoint = torch.cat(superpoint_list, 0).long()               # long[N]
        feats = torch.cat(feats, 0).float()                              # float (N, C)
        labels = torch.cat(labels, 0).long()                     # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()   # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)
        instance_center = torch.tensor(instance_center, dtype=torch.float32)
        instance_mask = torch.zeros((instance_pointnum.shape[0], len(labels)), dtype = torch.int) # int (total_nInst, N)
        for i in range(instance_mask.shape[0]): instance_mask[i] = (instance_labels == i).int()  
        #print(instance_center.shape, instance_center.sum(), instance_pointnum.shape[0])

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = dknet_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum, 'instance_mask': instance_mask,
                'instance_center': instance_center, 'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, "superpoint": superpoint}

    
    def testMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        superpoint_list = []
        superpoint_bias = 0
        labels = []
        instance_labels = []

        batch_offsets = [0]
        total_inst_num = 0
        #id = [i // cfg.test_batch_size for i in id]
        for i, idx in enumerate(id):
            if self.test_split == 'val' or self.test_split == 'train':
                # xyz_origin, rgb, label, instance_label = self.test_files[idx]
                xyz_origin, rgb, label, instance_label = self.read_sailvos3d(self.images_path, self.depth_path, self.rage_matrices_path, self.visible_path, self.labels_dict, idx)
                # for axis in range(3):
                #     plt.hist(xyz_origin[:,axis], bins=100)
                #     plt.savefig('/mnt/lustre/jkyang/PSG4D/dknet/vis/' + str(idx) + str(axis) + '.png')
                #     plt.close() 
                inst_num = instance_label.max() + 1
                instance_label[np.where(instance_label != -100)] += total_inst_num
                total_inst_num += inst_num

                labels.append(torch.from_numpy(label))
                instance_labels.append(torch.from_numpy(instance_label))
            elif self.test_split == 'test':
                xyz_origin, rgb = self.test_files[idx]
            else:
                print("Wrong test split: {}!".format(self.test_split))
                exit(0)

            # scene = self.test_file_names[idx].split('/')[-1][:12]
            # if not self.prefetch_superpoints:
            #     self.get_superpoint(scene, self.test_split)
            # superpoint = self.superpoints[scene]
            superpoint = np.zeros_like(label)
            ### flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)
            
            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            superpoint_list.append(torch.from_numpy(superpoint + superpoint_bias)) 
            superpoint_bias += (superpoint.max() + 1)

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))

        ### merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                         # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)           # float (N, 3)
        feats = torch.cat(feats, 0).float()                               # float (N, C)
        superpoints = torch.cat(superpoint_list, 0).long()               # long[N]

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = dknet_ops.voxelization_idx(locs, self.test_batch_size, self.mode)
        if self.test_split == 'val' or self.test_split == 'train':
            labels = torch.cat(labels, 0).long()                       # long (N)
            instance_labels = torch.cat(instance_labels, 0).long()   # long (N)
        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, "superpoint": superpoints}
    
    def read_sailvos3d(self, images_path, depth_path, rage_matrices_path, visible_path, labels_dict_all, idx):
        image = np.array(Image.open(images_path[idx]))
        
        depth = np.load(depth_path[idx])
        depth = depth/6.0 - 4e-5 # convert to NDC coordinate
        
        rage_matrices = np.load(rage_matrices_path[idx])
        # get the (ViewProj) matrix that transform points from the world coordinate to NDC
        # (points in world coordinate) @ VP = (points in NDC) 
        VP = rage_matrices['VP']
        VP_inverse = rage_matrices['VP_inv'] # NDC to world coordinate
        # get the (Proj) matrix that transform points from the camera coordinate to NDC
        # (points in camera coordinate) @ P = (points in NDC) 
        P = rage_matrices['P']
        P_inverse = rage_matrices['P_inv'] # NDC to camera coordinate
        
        H = 800
        W = 1280
        IMAGE_SIZE = (H, W)
        def pixels_to_ndcs(xx, yy, size=IMAGE_SIZE):
            s_y, s_x = size
            s_x -= 1  # so 1 is being mapped into (n-1)th pixel
            s_y -= 1  # so 1 is being mapped into (n-1)th pixel
            x = (2 / s_x) * xx - 1
            y = (-2 / s_y) * yy + 1
            return x, y
        
        px = np.arange(0, W)
        py = np.arange(0, H)
        px, py = np.meshgrid(px, py, sparse=False)
        px = px.reshape(-1)
        py = py.reshape(-1)

        ndcz = depth[py, px] # get the depth in NDC
        rgb = image[py, px, :]
        rgb = rgb / 127.5 - 1
        ndcx, ndcy = pixels_to_ndcs(px, py)
        ndc_coord = np.stack([ndcx, ndcy, ndcz, np.ones_like(ndcz)], axis=1)

        camera_coord = ndc_coord @ P_inverse
        camera_coord = camera_coord/camera_coord[:,-1:]

        world_coord = ndc_coord @ VP_inverse
        world_coord = world_coord/world_coord[:,-1:]
        # world_coord -= world_coord.min(0)

        # don't draw pixels with depth less than -100 (too far away, may be sky)
        depth_thre = np.sort(depth.reshape(-1))[int(world_coord.shape[0] * 0.3)]
        to_draw = depth >= depth_thre
        # print(images_path[idx], np.sum(to_draw))
        world_coord_fg = world_coord[to_draw.reshape(-1)]
        camera_coord_fg = camera_coord[to_draw.reshape(-1)]
        world_coord_fg -= world_coord_fg.min(0)
        camera_coord_fg -= camera_coord_fg.min(0)
        world_coord_fg = world_coord_fg[:,:-1]
        # resize_require = False
        # if world_coord_fg.max() > 30:
        #     print(world_coord_fg.max(0))
        #     world_coord_fg = world_coord_fg / world_coord_fg.max(0) * 30
        #     print(world_coord_fg.max(0))
        
        # for axis in range(3):
        #     if np.max(world_coord_fg[:,axis]) > 100:
        #         resize_require = True
        #         break
        # if resize_require:
        #     for axis in range(3):
        #         world_coord_fg[:,axis] = world_coord_fg[:,axis] / np.max(world_coord_fg[:,axis]) * 100
        #     print(images_path[idx], 'world: ', np.min(world_coord_fg[:,axis]), np.max(world_coord_fg[:,axis]))
        # for axis in range(3):
        #     print(images_path[idx], 'world: ', np.min(world_coord_fg[:,axis]), np.max(world_coord_fg[:,axis]))
            # print('camera: ', np.min(camera_coord[:,axis]), np.max(camera_coord[:,axis]))
        rgb_fg = rgb[to_draw.reshape(-1)]

        
        labels_instance = np.load(visible_path[idx])
        labels_instance[labels_instance == 0] = -100
        
        prefix = images_path[idx].split('/')[-3]
        labels_dict = labels_dict_all[prefix]
        labels_semantic = 25 * np.ones_like(labels_instance)
        for instance_id in np.unique(labels_instance):
            if instance_id != -100:
                if instance_id in labels_dict.keys():
                    labels_semantic[labels_instance == instance_id] = labels_dict[instance_id]
                else:
                    labels_instance[labels_instance == instance_id] = 24
        
        labels_instance_fg = labels_instance[to_draw.reshape(IMAGE_SIZE)]
        labels_semantic_fg = labels_semantic[to_draw.reshape(IMAGE_SIZE)]

        if world_coord_fg[::10].max() > 100:
            print('!!!!!!!!!!!', images_path[idx], world_coord_fg[::10].max(0), world_coord_fg[::10].min(0))
        
        return world_coord_fg[::10], rgb_fg[::10], labels_semantic_fg[::10], labels_instance_fg[::10]
        

if __name__ == "__main__":
    dataset = Dataset(test=False)
    dataset.trainLoader()
    dataloader = dataset.train_data_loader
    
    semantic_r_min = torch.zeros(20, dtype=torch.float).cuda()
    semantic_r_max = torch.zeros(20, dtype=torch.float).cuda()
    semantic_r_norm = torch.zeros(20, dtype=torch.float).cuda()
    semantic_ins_num = torch.zeros(20, dtype=torch.int).cuda()
    semantic_ins_size = torch.zeros(20, dtype=torch.int).cuda()
    
    for i, data in enumerate(dataloader):
        instance_info = data['instance_info'].cuda()
        labels = data['labels'].cuda()                        # (N), long, cuda
        instance_labels = data['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100
        instance_mask = data['instance_mask'].cuda()          # (total_nInst, N), int, cuda
        instance_size = data['instance_pointnum'].cuda()
        
        #print(instance_mask)
        instance_num = instance_labels.max() + 1
        r_min, _ = (instance_info[:, 0:3] - instance_info[:, 3:6]).min(1)
        r_max, _ = (instance_info[:, 0:3] - instance_info[:, 3:6]).max(1)
        r_norm = torch.norm((instance_info[:, 0:3] - instance_info[:, 3:6]), dim=1)
        _, ins_sample = instance_mask.max(1)
        instance_sem = labels[ins_sample]

        for i in range(instance_num):
            if instance_sem[i] != -100:
                semantic_r_min[instance_sem[i]] += r_min[i]
                semantic_r_max[instance_sem[i]] += r_max[i]
                semantic_r_norm[instance_sem[i]] += r_norm[i]
                semantic_ins_size[instance_sem[i]] += instance_size[i]
                semantic_ins_num[instance_sem[i]] += 1
        
        print(semantic_ins_num)
    
    print(semantic_r_min/(semantic_ins_num+1e-4))
    print(semantic_r_max/(semantic_ins_num+1e-4))
    print(semantic_r_norm/(semantic_ins_num+1e-4))
    print(semantic_ins_size/(semantic_ins_num+1e-4))
        