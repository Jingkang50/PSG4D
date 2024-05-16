from unitrack.test_mots_from_mask2former import eval_seq_3d
from mmcv import Config
import os
import glob

cfg = Config.fromfile('configs/unitrack/3d_hoi4d_trainset.py')

test_list_file = cfg.tracker_cfg.test_list_file
video_names = []
for line in open(test_list_file):
    video_names.append(line[:-1])
# print(video_names)

total_num_tubes_previous = 0
for video_name in video_names:
    print('Processing: ', video_name)
    num_tubes_this_video = eval_seq_3d(tracker_cfg=cfg.tracker_cfg, video_name=video_name, total_num_tubes_previous=total_num_tubes_previous)
    total_num_tubes_previous += num_tubes_this_video
    

