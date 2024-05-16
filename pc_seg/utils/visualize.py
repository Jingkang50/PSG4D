'''
Visualization
Written by Li Jiang
Modified by Wu Yizheng
'''

import numpy as np
import mayavi.mlab as mlab
import os, glob, argparse
import torch
from operator import itemgetter

COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

COLOR40 = np.array(
        [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168], [75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]])

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array(['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                        'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160]
}
SEMANTIC_IDX2NAME = {1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa', 7: 'table', 8: 'door', 9: 'window', 10: 'bookshelf', 11: 'picture',
                12: 'counter', 14: 'desk', 16: 'curtain', 24: 'refridgerator', 28: 'shower curtain', 33: 'toilet',  34: 'sink', 36: 'bathtub', 39: 'otherfurniture'}


def visualize_pts_rgb(fig, pts, rgb, scale=0.02):
    pxs = pts[:, 0]
    pys = pts[:, 1]
    pzs = pts[:, 2]
    pt_colors = np.zeros((pxs.size, 4), dtype=np.uint8)
    pt_colors[:, 0:3] = rgb
    pt_colors[:, 3] = 255  # transparent

    scalars = np.arange(pxs.__len__())
    points = mlab.points3d(pxs, pys, pzs,  scalars,
                           mode='sphere',  # point sphere
                           # colormap='Accent',
                           scale_mode='vector',
                           scale_factor=scale,
                           figure=fig)
    points.module_manager.scalar_lut_manager.lut.table = pt_colors


def sigmoid(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


def get_coords_color(opt):
    input_file = os.path.join(opt.data_root, opt.room_split, opt.room_name + '_inst_nostuff.pth')
    assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
    if opt.room_split == 'test':
        xyz, rgb = torch.load(input_file)
    else:
        xyz, rgb, label, inst_label = torch.load(input_file)
    rgb = (rgb + 1) * 127.5

    if (opt.task == 'semantic_gt'):
        assert opt.room_split != 'test'
        label = label.astype(int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif (opt.task == 'instance_gt'):
        assert opt.room_split != 'test'
        inst_label = inst_label.astype(int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

    elif (opt.task == 'semantic_pred'):
        assert opt.room_split != 'train'
        semantic_file = os.path.join(opt.result_root, opt.room_split, 'semantic', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif (opt.task == 'instance_pred'):
        assert opt.room_split != 'train'
        instance_file = os.path.join(opt.result_root, opt.room_split, opt.room_name + '.txt')
        assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)#np.ones(rgb.shape) * 100 #np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #

        for i in range(len(masks) - 1, -1, -1):
            #if i!=2: continue 
            mask_path = os.path.join(opt.result_root, opt.room_split, masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if (float(masks[i][2]) < 0.00):
                continue
            mask = np.loadtxt(mask_path).astype(int)
            print('{} {}: {} pointnum: {}'.format(i, masks[i], SEMANTIC_IDX2NAME[int(masks[i][1])], mask.sum()))
            if (mask.sum()==0 or SEMANTIC_IDX2NAME[int(masks[i][1])] == ''):# or SEMANTIC_IDX2NAME[int(masks[i][1])] != 'chair':
                continue
            inst_label_pred_rgb[mask == 1] = COLOR20[i % len(COLOR20)]
        rgb = inst_label_pred_rgb
    
    elif (opt.task == 'candidate'):
        candidate_file = os.path.join(opt.result_root, opt.room_split, 'candidate', opt.room_name + '.npy')
        candidate_score_file = os.path.join(opt.result_root, opt.room_split, 'candidate', opt.room_name + '_score.npy')
        candidate_r = np.ones(rgb.shape[0]) * 255
        candidate_g = np.ones(rgb.shape[0]) * 50
        candidate_b = np.ones(rgb.shape[0]) * 50
        
        candidate_pred = np.load(candidate_file).astype(int)
        candidate_score = np.load(candidate_score_file).astype(np.float) 
        candidate_score = softmax(candidate_score)

        scale = 0.3 / candidate_score[candidate_score!=0].mean()

        candidate_score = candidate_score *scale
        candidate_score = np.clip(candidate_score , a_min=0, a_max=1)
        candidate_r *= candidate_score
        candidate_r[candidate_r<=50] = 50
        
        
        candidate_rgb = np.append([candidate_r], [candidate_g], axis=0)
        candidate_rgb = np.append(candidate_rgb, [candidate_b], axis=0)
        candidate_rgb = np.swapaxes(candidate_rgb,0,1)
        candidate_rgb[candidate_pred] = COLOR40[candidate_pred % len(COLOR40)]
        
        rgb = candidate_rgb
        xyz = xyz
        label = label
        
        
    elif (opt.task == 'candidate_merge'):
        candidate_file = os.path.join(opt.result_root, opt.room_split, 'candidate', opt.room_name + '_merge.npy')
        candidate_score_file = os.path.join(opt.result_root, opt.room_split, 'candidate', opt.room_name + '_score.npy')
        candidate_r = np.ones(rgb.shape[0]) * 255
        candidate_g = np.ones(rgb.shape[0]) * 50
        candidate_b = np.ones(rgb.shape[0]) * 50
        
        candidate_pred = np.load(candidate_file).astype(int)
        candidate_score = np.load(candidate_score_file).astype(np.float) 
        scale = 1e-10 / candidate_score[candidate_score!=0].min()
        candidate_score = candidate_score *scale
        candidate_score = np.clip(candidate_score , a_min=0, a_max=1)
        candidate_r *= candidate_score
        candidate_r[candidate_r<=50] = 50
        
        
        candidate_rgb = np.append([candidate_r], [candidate_g], axis=0)
        candidate_rgb = np.append(candidate_rgb, [candidate_b], axis=0)
        candidate_rgb = np.swapaxes(candidate_rgb,0,1)
        candidate_rgb[candidate_pred] = COLOR40[candidate_pred % len(COLOR40)]
        
        rgb = candidate_rgb
        xyz = xyz
        label = label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='path to the input dataset files', default='../dataset/scannetv2')
    parser.add_argument('--result_root', help='path to the predicted results', default='exp/scannetv2/dknet/DKNet_run1_scannet/result/epoch400_scoret0.1_npointt100')
    parser.add_argument('--room_name', help='room_name', default='scene0011_00')
    parser.add_argument('--room_split', help='train / val / test', default='val')
    parser.add_argument('--task', help='input / semantic_gt / semantic_pred / instance_gt / instance_pred / candidate / candidate_merge', default='input')
    opt = parser.parse_args()

    print(opt.room_name)

    xyz, rgb = get_coords_color(opt)

    fig = mlab.figure(figure=None, bgcolor=(1.0, 1.0, 1.0), size=((800, 800)))
    visualize_pts_rgb(fig, xyz, rgb)
    mlab.show()

