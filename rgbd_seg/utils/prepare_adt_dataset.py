#11.7 convert video stream to mp4 ## Author: Choiszt
import numpy as np
import os
import sys
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from math import tan
import random 
import cv2
from tqdm import tqdm
from projectaria_tools import utils
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration
from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataProvider,
   AriaDigitalTwinSkeletonProvider,
   AriaDigitalTwinDataPathsProvider,
   bbox3d_to_line_coordinates,
   bbox2d_to_image_coordinates,
   utils as adt_utils,
)
from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from scipy import ndimage
import warnings
import json
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('paths_provider', type=str)
args = parser.parse_args()
paths_provider =f"./Aria_data/{args.paths_provider}"
try:
    gt_provider = AriaDigitalTwinDataPathsProvider(paths_provider)
except:
    print("killed!")
    os._exit(1)

def mkdir(SAVED_PATH):
    if not os.path.exists(SAVED_PATH):
        os.mkdir(SAVED_PATH)
SAVED_PATH=os.path.join("./data",args.paths_provider)
mkdir(SAVED_PATH)
selected_device_number = 0
data_paths = gt_provider.get_datapaths_by_device_num(selected_device_number,True) #if want to get occluded data,the args would be true

gt_provider = AriaDigitalTwinDataProvider(data_paths)
stream_id = StreamId("214-1")
img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(stream_id)
print("There are {} frames".format(len(img_timestamps_ns)))

with open(f"./instance_mapping/{args.paths_provider}.json","r")as f:
    instance_mapping=json.load(f)
check_path=lambda x:os.mkdir(x) if not os.path.exists(x) else None 

def map_value(x):
    if x != 0:
        return instance_mapping.get(str(x), x)
    return x

for iter in tqdm(range(len(img_timestamps_ns))):
    timestamp=img_timestamps_ns[iter]
    bbox_dict=gt_provider.get_object_2d_boundingboxes_by_timestamp_ns(timestamp, stream_id).data()
    temp_depth_provider=gt_provider.get_depth_image_by_timestamp_ns(timestamp, stream_id)
    temp_RGB_provider=gt_provider.get_aria_image_by_timestamp_ns(timestamp, stream_id)
    temp_SEG_provider = gt_provider.get_segmentation_image_by_timestamp_ns(timestamp, stream_id)
    raw_seg=temp_SEG_provider.data().to_numpy_array()

    vectorized_map = np.vectorize(map_value)
    raw_seg = vectorized_map(raw_seg)

    tempdata=np.repeat(temp_depth_provider.data().get_visualizable().to_numpy_array()[..., np.newaxis], 3, axis=2)
    tempdata_depth=temp_depth_provider.data().to_numpy_array() #TODO save to npy
    #tempdata_depth=temp_depth_provider.data().get_visualizable().to_numpy_array() #normalized to 0-255
    tempdata_rgb=np.repeat(temp_RGB_provider.data().to_numpy_array()[..., np.newaxis], 3, axis=2) if len(temp_RGB_provider.data().to_numpy_array().shape) < 3 else temp_RGB_provider.data().to_numpy_array()
    # tempSYN_rgb=np.repeat(temp_SYN_provider.data().to_numpy_array()[..., np.newaxis], 3, axis=2) if len(temp_SYN_provider.data().to_numpy_array().shape) < 3 else temp_SYN_provider.data().to_numpy_array()
    new_rgb=cv2.cvtColor(tempdata_rgb, cv2.COLOR_BGR2RGB)
    mkdir(os.path.join(SAVED_PATH,"image"))
    cv2.imwrite(os.path.join(SAVED_PATH,"image",f"{iter}.jpg"),new_rgb)
    mkdir(os.path.join(SAVED_PATH,"depth"))
    np.savez_compressed(os.path.join(SAVED_PATH,"depth",f"{iter}.npz"),**{"depth": tempdata_depth})
    mkdir(os.path.join(SAVED_PATH,"segmentation"))
    np.savez_compressed(os.path.join(SAVED_PATH,"segmentation",f"{iter}.npz"), **{"segmentation": raw_seg})
    #TODO imwrite the new_rgb to file
    #target_path=f"DATA/args.paths_provider/{iter}_rgb.jpg"
    #check_path(target_path) #check the path 是否存在
    #cv2.imwrite(target_path,new_rgb)

    #TODO imwrite the tempdata_depth to file
    #np.write(tempdata_depth)

print(f'Finish Writing {args.paths_provider}!')

