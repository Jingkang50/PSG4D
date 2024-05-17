# Building the PSG4D-HOI Dataset

This README provides instructions for building the PSG4D-HOI dataset.

## Step 1: Download the Data

1. Download the dataset files from https://hoi4d.github.io/. The dataset consists of the following ZIP files. To verify the integrity of the downloaded files, please check the provided MD5 checksums:

   | File Name                    | MD5 Checksum                     |
   |------------------------------|----------------------------------|
   | camera_params.zip            | 09f7e7ff630afbba022a2f524926bcc9 |
   | HOI4D_annotations.zip        | e68abb48478b4689a2f1380684bfcd68 |
   | HOI4D_depth_video.tar.gz0    | 34f9c031a97a0af5ab380e35879ea403 |
   | HOI4D_depth_video.tar.gz1    | 43c500ccb58c037e08fbce485e612030 |
   | HOI4D_depth_video.tar.gz2    | 0fddb54e2605898dc52e0ba868cb116b |
   | HOI4D_depth_video.tar.gz3    | 67dd7e3dc51eb5aff074cc6363213895 |
   | HOI4D_depth_video.tar.gz4    | 292121e662ae51e20b00e5d310d94f7b |
   | HOI4D_depth_video.tar.gz5    | 0db9b20aeabe67b8620ff1e123c54af1 |
   | HOI4D_depth_video.tar.gz6    | 0747aa07112a646e6ff7398ccc5816f2 |
   | HOI4D_release.zip            | c02d0c74935c7074f12c55ecb5f4d919 |
   
   To check the MD5 checksum of a file, you can use the following command:
   ```
   md5sum <filename>
   ```
   Make sure the computed checksums match the provided checksums.

2. If the checksums match and there are no issues with the downloaded files, proceed to extract the data by running the following command for the `HOI4D_depth_video.tar.gz` files:
   ```
   cat HOI4D_depth_video.tar.* | tar xvfz -
   ```
   This command concatenates all the tar.gz files and extracts their contents.

   For the ZIP files (`camera_params.zip`, `HOI4D_annotations.zip`, and `HOI4D_release.zip`), extract them using the appropriate ZIP extraction tool.


## Step 2: Organize the Dataset

Please check out [here](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EhUgIeYBPmVCvqeaJA-hmzkBdVcXt1QKtw3DX9a5zTnLsg?e=Rye0gd) and place the data into corresponding place.

```
- HOI
 - camera_params
 - HOI4D_annotations
   - ZY20210800001
   - ZY20210800002
   - ZY20210800003
   - ZY20210800004
 - HOI4D_depth_video
   - ZY20210800001
   - ZY20210800002
   - ZY20210800003
   - ZY20210800004
 - HOI4D_gaze
   - ZY20210800001
   - ZY20210800002
   - ZY20210800003
   - ZY20210800004
 - HOI4D_instructions
   - ZY20210800001
   - ZY20210800002
   - ZY20210800003
   - ZY20210800004
 - HOI4D_instructions
 - videos
   - zip
- categories.json
- genera_video.py
- hoi4d_id.json
- pag4d_hoi.json
- README.md
```


## Step 3: Visualize and Understand the Data

Run `generate_video.py` to see the video, mask, and scene graph annotations.

