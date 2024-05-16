import functools
import sys
import time

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append("../")
import backbone.unet as u
import data
import dknet_ops
from layers.layers import (Dynamic_weight_network_DFN, DynamicFilterLayer,
                           conv_with_kaiming_uniform)
from scipy.optimize import linear_sum_assignment
from utils import utils
from utils.config import cfg
from utils.log import logger

class DKNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_channel
        self.d = cfg.d
        d = self.d
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual
        self.DyWeight = cfg.DyWeight

        self.prepare_epochs = cfg.prepare_epochs
        self.semantic_epochs = cfg.semantic_epochs

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = 'ResidualBlock'
        else:
            block = 'VGGBlock'

        if cfg.use_coords:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, d, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = u.UBlock([d, 2*d, 3*d, 4*d, 5*d, 6*d, 7*d], norm_fn, block_reps, block, indice_key_id=1, add_transformer=cfg.add_transformer)

        self.output_layer = spconv.SparseSequential(
            norm_fn(d),
            nn.ReLU()
        )

        #### semantic branch
        self.linear = nn.Sequential(
            nn.Linear(d, d, bias=True),
            norm_fn(d),
            nn.ReLU(),
            nn.Linear(d, d, bias=True),
            norm_fn(d),
            nn.ReLU(),
            nn.Linear(d, classes, bias=True),
        )

        #### offset branch
        self.offset = nn.Sequential(
            nn.Linear(d, d, bias=True),
            norm_fn(d),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(d, 3, bias=True)

        #### score branch
        self.candidate_linear = nn.Sequential(
            nn.Linear(d+3, d, bias=True),
            norm_fn(d),
            nn.ReLU(),
            nn.Linear(d, d, bias=True),
            norm_fn(d),
            nn.ReLU(),
        )

        self.precision_linear = nn.Linear(d, 1, bias=True)
        self.recall_linear = nn.Linear(d, 1, bias=True)

        ################################
        ################################
        ################################
        ### for instance embedding
        self.output_dim = 16
        self.mask_conv_num = 3
        conv_block = conv_with_kaiming_uniform("BN", activation=True)

        mask_tower = []
        for i in range(self.mask_conv_num):
            mask_tower.append(conv_block(d, d))
        mask_tower.append(nn.Conv1d(
            d,  self.output_dim, 1
        ))
        self.add_module('mask_branch', nn.Sequential(*mask_tower))

        kernel_tower = []
        for i in range(self.mask_conv_num):
            kernel_tower.append(conv_block(d, d))
        kernel_tower.append(nn.Conv1d(
            d,  self.output_dim, 1
        ))
        self.add_module('kernel_branch', nn.Sequential(*kernel_tower))

        merge_tower = []
        for i in range(self.mask_conv_num):
            merge_tower.append(conv_block(2*self.output_dim+3, 2*self.output_dim+3))
        merge_tower.append(nn.Conv1d(
            2*self.output_dim+3,  1, 1
        ))
        self.add_module('merge_branch', nn.Sequential(*merge_tower))

        semantic_tower = []
        for i in range(self.mask_conv_num):
            semantic_tower.append(conv_block(self.output_dim, self.output_dim))
        semantic_tower.append(nn.Conv1d(
            self.output_dim, cfg.classes, 1
        ))
        self.add_module('semantic_branch', nn.Sequential(*semantic_tower))

        #### dynamic weight generator
        conv_shape = [16, 1]
        conv_dim = 0
        input_dim = self.output_dim + 3
        for dim in conv_shape:
            conv_dim += input_dim * dim + dim
            input_dim = dim

        self.weight_generator = Dynamic_weight_network_DFN(self.output_dim, conv_dim)
        self.DyConv = DynamicFilterLayer(conv_shape)

        self.apply(self.set_bn_init)

        #### fix parameter
        self.module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'linear': self.linear, 'offset': self.offset, 'offset_linear': self.offset_linear,
                       'mask': self.mask_branch, 'kernel': self.kernel_branch, 'merge': self.merge_branch,
                       'weight_generator': self.weight_generator}

        for m in self.fix_module:
            mod = self.module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        self.load_pretrain_module()

    def load_pretrain_module(self):
        #### load pretrain weights
        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path, map_location=torch.device('cpu'))['model']
            for m in self.pretrain_module:
                print("Load pretrained " + m + ": %d/%d" % utils.load_model_param(self.module_map[m], pretrain_dict, prefix=m))

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def topk_nms_d_sem_cuda(self, input, batch_offset, batch_idxs, coord, label, R=0.3, thres=0.3, local_thres=0.5, K=100):
        '''
        :param input: (N), float, cuda
        :param batch_offset: (B+1), int, cuda
        :param batch_idxs: (N), int, cuda
        :param coord: (N, 3), float, int
        :param label: (N), int, cuda
        :return:
        '''
        batch_offset = batch_offset.cpu()

        topk_idxs, sizes, k_foregrounds, k_backgrounds = dknet_ops.topk_nms(input, batch_offset, coord, label, R,
                    thres, local_thres, K, 2000, 2000)

        topk = input[topk_idxs].cuda()
        topk_idxs = topk_idxs.cuda()
        k_foregrounds = k_foregrounds.long().cuda()
        k_backgrounds = k_backgrounds.long().cuda()
        candidate_batch = batch_idxs[topk_idxs]
        sizes = sizes.float().cuda()

        return topk, topk_idxs, k_foregrounds, k_backgrounds, candidate_batch, sizes

    def candidate_merge(self, input_fg, input_bg, pos, batch_idx, epoch, merge_thre):
        '''
        :param input_fg: (N', d'), float, cuda
        :param input_bg: (N, d'), float, cuda
        :param pos: (N', 3), float, cuda
        :param batch_idx: (N'), float, int
        :param merge_thre: int, cuda
        :return: merge_pred: (I, I), float, cuda
        :return: ins_map: (I), int, cuda
        '''
        input_pos = torch.cat((input_fg, input_bg, pos), dim = 1) # I * (2d'+3)
        batch_mask = batch_idx.unsqueeze(1) - batch_idx.unsqueeze(0) # batch mask (I * I)

        input_diff = torch.clamp(torch.abs(input_pos.unsqueeze(1) - input_pos.unsqueeze(0)), min=1e-6) # I * I * (2d'+3)
        merge_pred = self.merge_branch(input_diff.permute(0,2,1)).permute(0,2,1).sigmoid() # I * I * 1
        merge_pred = torch.where(batch_mask.unsqueeze(-1)!=0, torch.zeros_like(merge_pred).cuda(), merge_pred)

        merge_score = merge_pred.clone()

        ins_map = torch.arange(input_fg.shape[0]) # (I), instance_id of each candidate

        if epoch > (cfg.prepare_epochs + cfg.semantic_epochs):
            ins_num = ins_map.shape[0] # I
            merge_score[torch.eye(ins_num).bool()] = 0

            # greedy aggregation
            while merge_score.max() > merge_thre:
                index = merge_score.argmax() # max score
                i = index // ins_num # candidate i
                j = index % ins_num # candidate j

                i_group = torch.where(ins_map[:] == ins_map[i])[0] # group i, candidates with the same instance_id as candidate i
                j_group = torch.where(ins_map[:] == ins_map[j])[0] # group j, candidates with the same instance_id as candidate j
                new_group = torch.cat((i_group, j_group), dim=0) # merged group

                new_group_h = new_group.view(-1, 1).repeat(1, new_group.shape[0])
                new_group_v = new_group.view(1, -1).repeat(new_group.shape[0], 1)
                merge_score[new_group_h, new_group_v] = 0 # set scores within the new group to 0

                ins_map_tmp = ins_map.clone()
                ins_map[new_group] = min(ins_map_tmp[i],ins_map_tmp[j]) # update ins_map

        return merge_pred, ins_map

    def instance_decoder(self, weights, mask_feats, clusters_ctr, coords, conv_masks, train=False):
        '''
        :param weight: (I, W), float, cuda
        :param mask_feats: (N, d'), float, cuda
        :param clusters_ctr: (I, 3), float, cuda
        :param coords: (N, 3), float, cuda
        :param conv_masks: (I, N), float, cuda
        :return: masks: (I, N), float, cuda
        '''
        point_num = coords.shape[0] # N

        if train:
            masks = torch.zeros((len(weights), point_num), device = mask_feats.device) # I * N
            for cluster in range(len(weights)):
                position_embedding = coords - clusters_ctr[cluster] # N * 3
                pts_feature = torch.cat((mask_feats, position_embedding), 1) # N * (d' + 3)
                #### convolution
                masks[cluster, conv_masks[cluster]] = self.DyConv((pts_feature[conv_masks[cluster]], weights[cluster])).view(-1) # N
        else:
            position_embedding = coords.unsqueeze(0) - clusters_ctr.unsqueeze(1) # I * N * 3
            mask_feats = torch.cat((mask_feats.unsqueeze(0).repeat(len(weights),1,1), position_embedding), 2) # I * N * (d' + 3)
            masks = self.DyConv((mask_feats, weights)) # I * N

        return masks.view(-1, point_num)

    def forward(self, input, input_map, coords, batch_idxs, epoch, ins_sample_num, train=True):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        '''
        ret = {}

        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]  # (N, d), float

        #### semantic branch
        semantic_scores = self.linear(output_feats)   # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1].long()    # (N), long

        ret['semantic_scores'] = semantic_scores

        #### offset branch
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats)   # (N, 3), float32

        ret['pt_offsets'] = pt_offsets

        if(epoch > self.semantic_epochs):
            #### valid points
            object_idxs = torch.nonzero(semantic_preds >= cfg.invalid_classes).view(-1)

            #### mask branch
            mask_feats = self.mask_branch(torch.unsqueeze(output_feats, dim=2).permute(2,1,0)).permute(2,1,0).squeeze()  # (N, d'), float

            batch_idxs_ = batch_idxs[object_idxs]
            coords_ = coords[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input.batch_size)
            semantic_preds_ = semantic_preds[object_idxs].int()

            #### predict center map
            candidate_feats = self.candidate_linear(torch.cat((output_feats[object_idxs], pt_offsets_), dim=1))

            precision_mask = self.precision_linear(candidate_feats)
            precision_mask_ = utils.batch_softmax(precision_mask, batch_idxs_, dim=0)

            candidate_score = precision_mask_

            #### candidate mining
            if not train:
                thres = cfg.test_thres
                local_thres = cfg.test_local_thres
            else:
                thres = cfg.train_thres
                local_thres = cfg.train_local_thres

            center_scores, candidate_pt, k_neighboors, k_backgrounds, candidate_batch, sizes = self.topk_nms_d_sem_cuda(
                candidate_score.detach(), batch_offsets_, batch_idxs_, coords_, semantic_preds_, 0.3, thres, local_thres, ins_sample_num)
            # center_scores: (N'), float
            # candidate_pt: (N'), int, point idxs of candidates in valid points
            # k_neighboors: (sumNPoints_f, 2), int, dim 0 for candidate_id, dim 1 for corresponding neighboor point idxs in valid points
            # k_backgrounds: (sumNPoints_b, 2), int, dim 0 for candidate_id, dim 1 for corresponding background point idxs in valid points
            # size: (N'), int, candidate size, the number of neighboor points

            instance_num = k_neighboors[-1, 0] + 1

            #### instance kernel generation
            kernel_feats = self.kernel_branch(torch.unsqueeze(output_feats, dim=2).permute(2,1,0)).permute(2,1,0).squeeze() # (N, d'), float
            kernel_feats_ = kernel_feats[object_idxs]

            kernel_fg_pooling = utils.neighboors_pooling(kernel_feats_, k_neighboors, instance_num) # (N', d'), float
            kernel_bg_pooling = utils.neighboors_pooling(kernel_feats_, k_backgrounds, instance_num)# (N', d'), float

            #### candidate aggregation
            merge_score, ins_map = self.candidate_merge(kernel_fg_pooling, kernel_bg_pooling,
                                (coords_ + pt_offsets_)[candidate_pt],candidate_batch, epoch, 0.5)
            # merge_score: (N', N'), float
            # ins_map:(N'), int, instance_id for each candidates

            merged_ins = torch.unique(ins_map) # (I), int
            candidate_pt_ = torch.zeros(merged_ins.shape[0]).cuda().long() # (I), int, point idxs of instance center in valid points
            kernel_merge = torch.zeros((merged_ins.shape[0], kernel_fg_pooling.shape[1])).cuda().float() # (I, d'), float, instance kernel
            candidate_batch_ = torch.zeros((merged_ins.shape[0])).cuda().long()
            for i in range(candidate_pt_.shape[0]):
                group = torch.where(ins_map == merged_ins[i])[0] # candidates in the same instance
                center = group[candidate_score[candidate_pt[group]].argmax()] # candidate with the highest center score is the instance center
                candidate_pt_[i] = candidate_pt[center]
                kernel_merge[i] = utils.weighted_mean(sizes[group], kernel_fg_pooling[group]) # size weighted average
                candidate_batch_[i] = candidate_batch[center]

            _ , batch_count = torch.unique(candidate_batch, return_counts=True)

            if train and candidate_pt_.shape[0]>cfg.MAX_INST_NUM:
                candidate_pt_ = candidate_pt_[:cfg.MAX_INST_NUM]
                kernel_merge = kernel_merge[:cfg.MAX_INST_NUM]
                candidate_batch_ = candidate_batch_[:cfg.MAX_INST_NUM]

            #### instance decoding by dynamic convolution
            if epoch > (cfg.prepare_epochs + cfg.semantic_epochs):
                weights = self.weight_generator(kernel_merge) # (I, W), W is up to kernel dim and num
                print(weights.shape, kernel_merge.shape)
                batch_mask = (batch_idxs.repeat(input.batch_size, 1) == torch.arange(input.batch_size).cuda().unsqueeze(-1))
                ### instance decoding
                masks = self.instance_decoder(weights, mask_feats, coords_[candidate_pt_], coords, batch_mask[candidate_batch_], train) #(I, N)

                ### calculate thres for each instance
                thres = dknet_ops.otsu(masks, 100)
                thre_masks = torch.where(masks < thres.unsqueeze(-1), torch.zeros_like(masks).cuda(), masks)
                seg_score, seg_result = thre_masks.max(0)
                if torch.sum(thres != 0):
                    seg_result[seg_score < thres[thres != 0].min()] = -100 # (N), hard segmentation result

                ### voting within the predicted instance to obtain semantic labels
                ins_sem_preds = utils.get_instance_seg_pred_label(semantic_preds_, seg_result[object_idxs], masks.shape[0]) #(I), 0-nClass, nClass for invalid instance

                ret['decoding'] = masks, thres, thre_masks, seg_result, ins_sem_preds, weights, kernel_merge

            ret['candidate'] = (object_idxs[candidate_pt], precision_mask_.squeeze())
            ret['merge'] = (object_idxs[candidate_pt_], merge_score, (batch_count**2).sum())

        return ret

def loss_fn(loss_inp, epoch):
    loss_out = {}
    ### criterion
    seg_num_per_class = torch.Tensor([18422, 2066, 35, 188, 3.5, 42, 34, 6.5, 24, 20, 9.3, 413, 99, 85, 46, 71, 1, 5, 28, 40, 420, 788, 29, 8.7, 1, 23605])
    seg_labelweights = seg_num_per_class / torch.sum(seg_num_per_class)
    seg_labelweights = torch.Tensor(torch.pow(torch.max(seg_labelweights) / seg_labelweights, 1 / 3.0)).cuda()
    # pixel_sem_criterion = nn.CrossEntropyLoss(weight=seg_labelweights, ignore_index=cfg.ignore_label)
    pixel_sem_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    merge_criterion = nn.BCELoss(reduction='none').cuda()
    mask_criterion = nn.BCELoss(reduction='none').cuda()

    '''semantic loss'''
    semantic_scores, semantic_labels = loss_inp['semantic_scores']
    semantic_preds = semantic_scores.max(1)[1].long()  # (N), long
    # semantic_scores: (N, nClass), float32, cuda
    # semantic_labels: (N), long, cuda
    # CELoss
    object_idxs = torch.nonzero(semantic_preds >= cfg.invalid_classes).view(-1)
    semantic_loss = pixel_sem_criterion(semantic_scores, semantic_labels)
    # multi-classes dice loss
    semantic_labels_ = semantic_labels[semantic_labels != cfg.ignore_label]
    semantic_scores_ = semantic_scores[semantic_labels != cfg.ignore_label]
    one_hot_labels = F.one_hot(semantic_labels_, num_classes=cfg.classes)
    semantic_scores_softmax = F.softmax(semantic_scores_, dim=-1)
    semantic_loss += utils.dice_loss_multi_classes(semantic_scores_softmax, one_hot_labels).mean()
    # semantic accuracy
    semantic_acc = ((semantic_labels == semantic_preds)[semantic_labels!=-100]).float().sum() / (semantic_labels!=-100).sum()

    loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])
    mask_label, coords, instance_info, instance_labels, pt_offsets = loss_inp['mask_label']

    valid = (instance_labels != cfg.ignore_label).float()
    '''offsets loss'''
    gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
    pt_diff = pt_offsets - gt_offsets   # (N, 3)
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
    pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
    pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
    direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
    offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

    offset_loss = offset_norm_loss + offset_dir_loss
    loss_out['offset_loss'] = (offset_loss, valid.sum())

    if(epoch > cfg.semantic_epochs):
        candidate_pt, candidate_score, batch_idxs = loss_inp['candidate_pt']
        merged_candidate, merge_score, merge_num = loss_inp['merge']

        _, gt_sample = torch.max(mask_label, dim=1)
        gt_sample = gt_sample.unsqueeze(1)

        '''center map loss'''
        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        gt_dist = torch.norm(gt_offsets[object_idxs], dim=1)
        gt_r, _ = (instance_info[object_idxs, 6:9] - instance_info[object_idxs, 3:6]).max(1)
        candidate_mask_gt = torch.exp(-25*((gt_dist**2)/((gt_r+1e-4)**2))) # (N)
        sample_loss = torch.sum(torch.abs(candidate_score - candidate_mask_gt)* valid[object_idxs]) / (torch.sum(valid[object_idxs]) + 1e-6)

        '''merge loss'''
        merge_gt = torch.zeros_like(merge_score)
        for i in range(merge_score.shape[0]):
            ins = torch.where(mask_label[:, candidate_pt[i]] == 1)[0]
            if ins.shape[0] == 0: continue
            merge_gt[i] = mask_label[ins, candidate_pt].unsqueeze(-1)

        merge_loss = merge_criterion(merge_score, merge_gt).sum() / merge_num
        merge_loss += utils.diceLoss(merge_score, merge_gt, 1e-6).mean()

        if epoch > (cfg.prepare_epochs + cfg.semantic_epochs):
            '''mask loss'''
            mask_preds, clusters_category, result = loss_inp['masks_preds']
            batch_mask = (batch_idxs.repeat(cfg.batch_size, 1) == torch.arange(cfg.batch_size).cuda().unsqueeze(-1))
            # mask_pred: (nProposal, N) prediction of masks
            # mask_label: (total_nInst, N) ground truth of masks

            # Hungarian algorithm
            candidate_cost = torch.norm((instance_info[gt_sample, 0:3] - coords[merged_candidate].unsqueeze(0)), dim=-1) # center distance
            candidate_cost += 1 * ((semantic_labels[gt_sample].cuda() - clusters_category.unsqueeze(0)) != 0).float() # semantic mask
            candidate_cost += 100 * ((batch_idxs[gt_sample].cuda() - batch_idxs[merged_candidate].unsqueeze(0)) != 0).float() # batch mask
            # candidate_cost, (I, G), float, I ins pred num, G ins gt num, alignmeant cost
            row_ind, col_ind = linear_sum_assignment(candidate_cost.cpu())
            row_ind, col_ind = torch.Tensor(row_ind).cuda().long(), torch.Tensor(col_ind).cuda().long()
            row_ind = row_ind % gt_sample.shape[0]

            assign_cost = candidate_cost[row_ind, col_ind]
            cost_mask = (assign_cost < 100) # batch mask
            if cost_mask.sum()>0:
                row_ind = row_ind[cost_mask]
                col_ind = col_ind[cost_mask]

            col_ind_sorted, indices = torch.sort(col_ind)
            row_ind_sorted = row_ind[indices]

            # matched pred and gt
            mask_preds = mask_preds[col_ind_sorted]
            merged_candidate = merged_candidate[col_ind_sorted]
            mask_gt = torch.index_select(mask_label.float(), 0, row_ind_sorted) # (nProposal, N)
            ins_sem_gt = semantic_labels[gt_sample][row_ind_sorted].squeeze()

            # iou
            masks_result = torch.zeros_like(mask_preds).cuda()
            for i in range(masks_result.shape[0]): masks_result[i] = (result == col_ind_sorted[i]).int()
            inter = (masks_result*mask_gt).sum(1)
            point_num = masks_result.sum(1) + mask_gt.sum(1)
            ious = inter / (point_num - inter + 1e-6)

            # valid matches
            valid_ins = (ins_sem_gt != -100) & (ious > 0.25)
            if valid_ins.sum() == 0: valid_ins = (ins_sem_gt != -100)
            mask_preds = mask_preds[valid_ins]
            mask_gt = mask_gt[valid_ins]
            merged_candidate = merged_candidate[valid_ins]

            # instance semantic accuracy
            ins_sem_acc = (clusters_category[col_ind_sorted] == ins_sem_gt)[valid_ins].sum().float() / (valid_ins + 1e-6).sum()

            # mask loss
            mask_loss = (mask_criterion(mask_preds.view(-1), mask_gt.view(-1)))[batch_mask[batch_idxs[merged_candidate].long()].view(-1)].mean()
            dice_loss = utils.diceLoss(mask_preds, mask_gt, 1e-6)
            mask_loss += dice_loss.mean()

            loss_out['mask_loss'] = (mask_loss, valid_ins.sum())


    loss = cfg.loss_weight[0] * semantic_loss + cfg.loss_weight[1] * offset_loss
    if epoch > (cfg.prepare_epochs + cfg.semantic_epochs):
        loss += cfg.loss_weight[2] * mask_loss + cfg.loss_weight[3] * sample_loss + cfg.loss_weight[3] * merge_loss
        print('foreground acc:{}, instance semantic acc:{}, ins num:{:03d}'.format(semantic_acc.item(), ins_sem_acc, int(valid_ins.sum())))
        print('semantic_loss: {}, offset_loss:{}, mask_loss: {}, sample_loss: {}, merge_loss: {}'.format(semantic_loss, offset_loss,  mask_loss, sample_loss, merge_loss))
    elif (epoch > cfg.semantic_epochs):
        loss += cfg.loss_weight[3] * sample_loss + cfg.loss_weight[3] * merge_loss
        print('foreground acc:{}'.format(semantic_acc.item()))
        print('semantic_loss: {}, offset_loss:{}, sample_loss: {}, merge_loss: {}'.format(semantic_loss, offset_loss, sample_loss, merge_loss))
    else:
        print('foreground acc:{}'.format(semantic_acc.item()))
        print('semantic_loss:{}, offset_loss:{}'.format(semantic_loss, offset_loss))

    return loss, loss_out

def train_fn(model, data, epoch):
    coords = data['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
    voxel_coords = data['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
    p2v_map = data['p2v_map'].cuda()                      # (N), int, cuda
    v2p_map = data['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda
    coords_float = data['locs_float'].cuda()              # (N, 3), float32, cuda
    feats = data['feats'].cuda()                          # (N, C), float32, cuda
    labels = data['labels'].cuda()                        # (N), long, cuda
    instance_labels = data['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100

    instance_info = data['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
    instance_mask = data['instance_mask'].cuda()          # (total_nInst, N), int, cuda

    spatial_shape = data['spatial_shape']

    if cfg.use_coords:
        feats = torch.cat((feats, coords_float), 1).float()
    voxel_feats = dknet_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

    input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), torch.tensor(spatial_shape).cuda(), coords[:, 0].int().max()+1)
    print(input_.features.shape, p2v_map.shape, coords_float.shape, coords[:, 0].int().shape, torch.tensor(epoch, dtype=int).unsqueeze(0).shape,  input_.batch_size.unsqueeze(0).shape)
    ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), torch.tensor(epoch, dtype=int).unsqueeze(0).cuda(),  input_.batch_size.unsqueeze(0)*150)

    if ret == None:
        return None
    semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
    pt_offsets = ret['pt_offsets']           # (N, 3), float32, cuda
    if(epoch > cfg.semantic_epochs):
        candidate_pt, candidate_score = ret['candidate']
        merged_candidate, merge_score, merge_num = ret['merge']
        if epoch > (cfg.prepare_epochs + cfg.semantic_epochs):
            masks, thres, thre_masks, seg_result, ins_sem_preds,  weights, kernel_merge = ret['decoding']

    loss_inp = {}
    loss_inp['semantic_scores'] = (semantic_scores, labels)
    loss_inp['mask_label'] = (instance_mask, coords_float, instance_info, instance_labels, pt_offsets)
    if(epoch > cfg.semantic_epochs):
        loss_inp['candidate_pt'] = (candidate_pt, candidate_score, coords[:, 0].int())
        loss_inp['merge'] = merged_candidate, merge_score, merge_num
        if epoch > (cfg.prepare_epochs + cfg.semantic_epochs):
            loss_inp['masks_preds'] = (masks, ins_sem_preds, seg_result)

    loss, loss_out = loss_fn(loss_inp, epoch)

    ##### accuracy / visual_dict / meter_dict
    with torch.no_grad():
        preds = {}
        preds['pt_offsets'] = pt_offsets
        preds['semantic'] = semantic_scores
        if (epoch > cfg.semantic_epochs):
            preds['candidate'] = (candidate_pt, candidate_score)
            preds['merge'] = ret['merge']
            if epoch > (cfg.prepare_epochs + cfg.semantic_epochs):
                preds['masks'] = (masks, ins_sem_preds)

        visual_dict = {}
        visual_dict['loss'] = loss
        for k, v in loss_out.items():
            visual_dict[k] = v[0]

        meter_dict = {}
        meter_dict['loss'] = (loss.item(), coords.shape[0])
        for k, v in loss_out.items():
            meter_dict[k] = (float(v[0]), v[1])

    return loss, preds, visual_dict, meter_dict

def test_fn(data, model, epoch):
    coords = data['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
    voxel_coords = data['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
    p2v_map = data['p2v_map'].cuda()          # (N), int, cuda
    v2p_map = data['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

    coords_float = data['locs_float'].cuda()  # (N, 3), float32, cuda
    feats = data['feats'].cuda()              # (N, C), float32, cuda

    spatial_shape = data['spatial_shape']

    if cfg.use_coords:
        feats = torch.cat((feats, coords_float), 1)
    voxel_feats = dknet_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

    input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.test_batch_size)
    # print(input_.features, p2v_map, coords_float, coords[:, 0].int(), torch.tensor(epoch, dtype=int).unsqueeze(0),  torch.tensor(input_.batch_size, dtype=int).unsqueeze(0))
    ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), torch.tensor(epoch, dtype=int).unsqueeze(0),  torch.tensor(input_.batch_size, dtype=int).unsqueeze(0)*150, False)
    # ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), epoch, 150 * cfg.test_batch_size, False)
    if ret == None:
        return None
    semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
    pt_offsets = ret['pt_offsets']           # (N, 3), float32, cuda
    if(epoch > cfg.semantic_epochs):
        candidate_pt, candidate_score = ret['candidate']
        merged_candidate, merge_score, merge_num = ret['merge']
        if epoch > (cfg.prepare_epochs + cfg.semantic_epochs):
            masks, thres, thre_masks, seg_result, ins_sem_preds, weights, kernel_merge = ret['decoding']

    ##### preds
    with torch.no_grad():
        preds = {}
        preds['semantic'] = semantic_scores
        preds['pt_offsets'] = pt_offsets
        if (epoch > cfg.semantic_epochs):
            #preds['clusters_idx'] = ret['clusters_idx']
            preds['candidate'] = (candidate_pt, candidate_score)
            preds['decoding'] = (masks, thres, thre_masks, seg_result, ins_sem_preds, weights, kernel_merge)
            preds['merge'] = merged_candidate, merge_score, merge_num

    return preds


if __name__ == "__main__":
    am_dict = {}

    model = DKNet(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model.cuda()

    dataset = data.scannetv2_inst.Dataset()
    dataset.trainLoader()
    dataset.valLoader()
    trainloader = dataset.train_data_loader

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    ##### train and val
    for epoch in range(0, cfg.epochs + 1):
        model.train()
        start_epoch = time.time()

        for i, data in enumerate(trainloader):
            torch.cuda.empty_cache()

            ##### adjust learning rate
            utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)

            ##### prepare input and forward
            loss, _, visual_dict, meter_dict = train_fn(model, data, epoch)

            ##### meter_dict
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                am_dict[k].update(v[0], v[1])

            ##### backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write(
                "epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f})\n".format
                (epoch, cfg.epochs, i + 1, len(trainloader), am_dict['loss'].val, am_dict['loss'].avg))
            if (i == len(trainloader) - 1): print()

        logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

        utils.checkpoint_save(model, optimizer, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, cfg.use_cuda)
