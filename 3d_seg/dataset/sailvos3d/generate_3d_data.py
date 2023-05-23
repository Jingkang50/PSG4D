import os
import numpy as np
import json
from PIL import Image
import torch
import time
import multiprocessing as mp

objects_all =['sky', 'floor', 'ceiling', 'ground', 'wall', 'grass', 'fence', 'stair', 'window', "other object",
              'chair', 'sofa', 'table', 'bed', "person", "car", "motorcycle", "truck", "bird", 
              "dog", "handbag", "suitcase", "bottle", "cup", "bowl", "potted plant", 
              "dining table", "tv", "laptop", "cell phone", "bag", "bin", "box", "door", 
              "road barrier", "stick"]
labels_dict = {}
num_valid = 0
with open('/mnt/lustre/jkyang/PSG4D/share/GTA/psg4d_background.json', 'r') as fcc_file:
        labels_json_file = json.load(fcc_file)
        object_to_cls = {}
        for obj in objects_all:
            object_to_cls[obj] = objects_all.index(obj)
        for k, dict_tmp in labels_json_file.items():
            object_dict_new = {}
            for objects_dict in dict_tmp['object']:
                object_dict_new[objects_dict['object_id']] = object_to_cls[objects_dict['category']]
            labels_dict[dict_tmp['video_id']] = object_dict_new
            # print(dict_tmp['video_id'], object_dict_new)
            num_valid += 1

rage_matrices_path = "/mnt/lustre/jkyang/PSG4D/share/GTA/rage_matrices"
g = os.walk(rage_matrices_path)
rage_files_all = []
for path,dir_list,file_list in g:  
    for file_name in file_list:  
        rage_files_all.append(os.path.join(path, file_name))
print(len(rage_files_all))
fn = rage_files_all
# prefix = fn[:35]
# id_num = fn[-10:-3]
# video_name = fn.split('/')[8] + '/'
# images_path = prefix + "images/" + video_name + id_num + "bmp"
# depth_path = prefix + "depth/" + video_name + id_num + "npy"
# visible_path = prefix + "visible/" + video_name + id_num + "npy"
# print(prefix, id_num, images_path, depth_path, visible_path)
total_num = 0
valid_num = 0

def f(fn):
    prefix = fn[:35]
    id_num = fn[-10:-3]
    video_name = fn.split('/')[8] + '/'
    images_path = prefix + "images/" + video_name + id_num + "bmp"
    depth_path = prefix + "depth/" + video_name + id_num + "npy"
    visible_path = prefix + "visible/" + video_name + id_num + "npy"

    image = np.array(Image.open(images_path))
        
    depth = np.load(depth_path)
    depth = depth/6.0 - 4e-5 # convert to NDC coordinate
    
    rage_matrices = np.load(fn)
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
    # print(images_path, np.sum(to_draw))
    world_coord_fg = world_coord[to_draw.reshape(-1)]
    camera_coord_fg = camera_coord[to_draw.reshape(-1)]
    world_coord_fg -= world_coord_fg.min(0)
    camera_coord_fg -= camera_coord_fg.min(0)
    world_coord_fg = world_coord_fg[:,:-1]
    rgb_fg = rgb[to_draw.reshape(-1)]

    labels_instance = np.load(visible_path)
    labels_instance[labels_instance == 0] = -100
    
    prefix = images_path.split('/')[-3]
    global labels_dict
    labels_dict_tmp = labels_dict[video_name[:-1]]
    labels_semantic = 25 * np.ones_like(labels_instance)
    for instance_id in np.unique(labels_instance):
        if instance_id != -100:
            if instance_id in labels_dict_tmp.keys():
                labels_semantic[labels_instance == instance_id] = labels_dict_tmp[instance_id]
            else:
                labels_semantic[labels_instance == instance_id] = 9 # other object
    
    labels_instance_fg = labels_instance[to_draw.reshape(IMAGE_SIZE)]
    labels_semantic_fg = labels_semantic[to_draw.reshape(IMAGE_SIZE)]

    save_path = '/mnt/lustre/jkyang/PSG4D/share/dataset_jcenaa/dataset/sailvos3d_2/data_3d/' + video_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    if np.max(world_coord_fg) <= 10:
        torch.save((world_coord_fg[::10], rgb_fg[::10], labels_semantic_fg[::10], labels_instance_fg[::10]), save_path + prefix + "_" + id_num + 'pth')
        print('Saving to ' + save_path + video_name[:-1] + "_" + id_num + 'pth')
        global valid_num
        valid_num += 1
    else:
        print('Not Saving to ' + save_path + video_name[:-1] + "_" + id_num + 'pth')
    global total_num
    total_num += 1

begin = time.time()
p = mp.Pool(processes=16)
p.map(f, fn)
p.close()
p.join()
print(total_num, valid_num)
print(time.time() - begin)