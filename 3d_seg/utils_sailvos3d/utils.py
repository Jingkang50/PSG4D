import glob
import os
import sys
from math import cos, pi

import numpy as np
import torch

sys.path.append('../')

from typing import Optional

from scipy.sparse import coo_matrix

from utils.config import cfg
from utils.log import logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr =  clip + 0.5 * (base_lr - clip) * \
            (1 + cos(pi * ( (epoch - step_epoch) / (total_epochs - step_epoch))))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))  # area_intersection: K, indicates the number of members in each class in intersection
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def checkpoint_restore(model, optimizer, exp_path, exp_name, use_cuda=True, epoch=0, dist=False, f=''):
    if use_cuda:
        model.cpu()

    if not f:
        if epoch > 0:
            f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
            assert os.path.isfile(f)
        else:
            f = sorted(glob.glob(os.path.join(exp_path, exp_name + '-*.pth')))
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path) + len(exp_name) + 2 : -4])

    if len(f) > 0:
        logger.info('Restore from ' + f)

        checkpoint = torch.load(f, map_location=torch.device('cpu'))

        if dist:
            torch.distributed.barrier()

        model_dict = checkpoint['model']
        optimizer_dict = checkpoint['optimizer']
        #print(model_dict)
        for k, v in model_dict.items():
            if 'module.' in k:
                model_dict = {k[len('module.'):]: v for k, v in model_dict.items()}
            break
        #print(model_dict)
        if dist:
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict, strict=True)
        #print(optimizer_dict)
        if optimizer != None:
            optimizer.load_state_dict(optimizer_dict)
            for state in optimizer.state.values():
                if state == None: continue
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        if dist:
            torch.distributed.barrier()

    if use_cuda:
        model.cuda()

    return epoch + 1

def ap_restore(exp_path, exp_name):
    f = glob.glob(os.path.join(exp_path, exp_name + 'ap*.pth'))
    ap, ap_50, ap_25 = 0, 0, 0
    if len(f) == 0:
        return 0, 0, 0
    else:
        for fname in f:
            if 'ap50' in fname:
                ap_50_tmp = float(fname.split('.')[0].split('_')[-1]) / 100
                if ap_50 < ap_50_tmp: ap_50 = ap_50_tmp
            elif 'ap25' in fname:
                ap_25_tmp = float(fname.split('.')[0].split('_')[-1]) / 100
                if ap_25 < ap_25_tmp: ap_25 = ap_25_tmp
            else:
                ap_tmp = float(fname.split('.')[0].split('_')[-1]) / 100
                if ap < ap_tmp: ap = ap_tmp

        return ap, ap_50, ap_25

def get_start_epoch(exp_path, exp_name, epoch=0, f=''):
    if not f:
        if epoch > 0:
            f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
            assert os.path.isfile(f)
        else:
            f = sorted(glob.glob(os.path.join(exp_path, exp_name + '-*.pth')))
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path) + len(exp_name) + 2 : -4])

    return epoch + 1

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def checkpoint_save(model, optimizer, exp_path, exp_name, epoch, save_freq=16, use_cuda=True, dist=False):
    f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    logger.info('Saving ' + f)
    model.cpu()
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }, f)
    if use_cuda:
        model.cuda()

    #remove previous checkpoints unless they are a power of 2 or a multiple of 16 to save disk space
    epoch = epoch - 1
    f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(f)


def load_model_param(model, pretrained_dict, prefix=""):
    # suppose every param in model should exist in pretrain_dict, but may differ in the prefix of the name
    # For example:    model_dict: "0.conv.weight"     pretrain_dict: "FC_layer.0.conv.weight"
    model_dict = model.state_dict()
    len_prefix = 0 if len(prefix) == 0 else len(prefix) + 1
    pretrained_dict_filter = {k[len_prefix:]: v for k, v in pretrained_dict.items() if k[len_prefix:] in model_dict and prefix in k}
    assert len(pretrained_dict_filter) > 0
    model_dict.update(pretrained_dict_filter)
    model.load_state_dict(model_dict)
    return len(pretrained_dict_filter), len(model_dict)


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()


def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def print_error(message, user_fault=False):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    if user_fault:
      sys.exit(2)
    sys.exit(-1)

def diceLoss(mask_pred, mask_gt, ep=1e-8):
    inter = 2 * (mask_gt * mask_pred).sum(1) + 1
    union = (mask_gt ** 2.0).sum(1) + (mask_pred ** 2.0).sum(1) + 1 + ep
    dice_loss = 1 - inter / union

    return dice_loss

def dice_loss_multi_classes(input: torch.Tensor,
                            target: torch.Tensor,
                            epsilon: float = 1e-5,
                            weight: Optional[float]=None) -> torch.Tensor:
    r"""
    modify compute_per_channel_dice from https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # convert the feature channel(category channel) as first
    axis_order = (1, 0) + tuple(range(2, input.dim()))
    input = input.permute(axis_order)
    target = target.permute(axis_order)

    target = target.float()
    # Compute per channel Dice Coefficient
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / \
                       (torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)

    loss = 1. - per_channel_dice

    return loss

def weighted_mean(weights, input):
    return (weights.unsqueeze(-1)*input).sum(0) / weights.sum()

def batch_softmax(input, batch_idxs, dim=0):
    batch_size = batch_idxs.max() + 1

    for batch in range(batch_size):
        batch_mask = batch_idxs == batch
        input[batch_mask] = torch.softmax(input[batch_mask], dim=0)

    return input

def get_instance_seg_pred_label(semantic_label, seg_result, instance_num=0):
    if instance_num == 0:
        instance_num = seg_result.max() + 1
    seg_labels = []
    for n in range(instance_num):
        mask = (seg_result == n)
        if mask.sum() == 0:
            seg_labels.append(cfg.classes)
            continue
        seg_label_n = torch.mode(semantic_label[mask])[0].item()
        seg_labels.append(seg_label_n)

    return torch.Tensor(seg_labels).cuda()

def align_superpoint_label(labels: torch.Tensor,
                           superpoint: torch.Tensor,
                           num_label: int=20,
                           ignore_label: int=-100):
    r"""refine semantic segmentation by superpoint

    Args:
        labels (torch.Tensor, [N]): semantic label of points
        superpoint (torch.Tensor, [N]): superpoint cluster id of points
        num_label (int): number of valid label categories
        ignore_label (int): the ignore label id

    Returns:
        label: (torch.Tensor, [num_superpoint]): superpoint's label
        label_scores: (torch.Tensor, [num_superpoint, num_label + 1]): superpoint's label scores
    """
    row = superpoint.cpu().numpy() # superpoint has been compression
    col = labels.cpu().numpy()
    col[col < 0] = num_label
    data = np.ones(len(superpoint))
    shape = (len(np.unique(row)), num_label + 1)
    label_map = coo_matrix((data, (row, col)), shape=shape).toarray()  # [num_superpoint, num_label + 1]
    label = torch.Tensor(np.argmax(label_map, axis=1)).long().to(labels.device)  # [num_superpoint]
    label[label == num_label] = ignore_label # ignore_label
    label_scores = torch.Tensor(label_map.max(1) / label_map.sum(axis=1)).to(labels.device) # [num_superpoint, num_label + 1]

    return label, label_scores

def neighboors_pooling(input, masks, instance_num=0):
    if instance_num == 0:
        instance_num = masks[-1, 0] + 1

    input_mean = input.mean(0).detach()
    pooling_output = torch.zeros((instance_num, input.shape[1])).cuda()
    for i in range(instance_num):
        mask = masks[masks[:, 0] == i, 1]
        if mask.shape[0] == 0:
            pooling_output[i] = input_mean
        else:
            mask_input = input[mask]
            maxpooling = mask_input.mean(0)
            pooling_output[i] = maxpooling

    return pooling_output
