"""
This script processes video data and reorganizes it into a new directory structure.
It reads video cut information from 'video_cut.json' and iterates over each video ID and frame range.
For each frame, it copies the corresponding data files (camera, depth, images, rage_matrices, visible)
from the source directory './sailvos3d/' to the destination directory './psg4d_gta/'.
The script uses tqdm to display progress bars for both video and frame processing.
"""

import os
import json
import shutil
from tqdm import tqdm

with open('video_cut.json', 'r') as f:
    video_cut = json.load(f)

data_types = ['camera', 'depth', 'images', 'rage_matrices', 'visible']
file_types = ['yaml', 'npy', 'bmp', 'npz', 'npy']

for video_id, cut in tqdm(video_cut.items(), desc='Processing videos', unit='video'):
    for new_idx, old_idx in tqdm(enumerate(range(cut[0], cut[1] + 1)), \
        desc=f'Processing frames for {video_id}', unit=f'/{cut[1] - cut[0] + 1} frame', leave=False):
        for data_type, file_type in zip(data_types, file_types):
            src_pth = f'./sailvos3d/{video_id}/{data_type}/{old_idx:06d}.{file_type}'
            dst_pth = f'./psg4d_gta/{data_type}/{video_id}/{new_idx:06d}.{file_type}'
            os.makedirs(f'./psg4d_gta/{data_type}/{video_id}', exist_ok=True)
            if os.path.exists(src_pth):
                shutil.copy(src_pth, dst_pth)
            else:
                print(f'{src_pth} not found')
