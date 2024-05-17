# Setting up PSG4D-GTA Dataset

This README provides instructions for downloading and setting up the PSG4D-GTA dataset.

## Step 1: Download the Data

1. The PSG4D-GTA dataset uses the data from [SAIL-VOS 3D dataset](https://sailvos.web.illinois.edu/_site/_site/index.html). Please contact SAIL-VOS 3D's authors for data. 
Based on the SAIL-VOS 3D dataset, we filter and cut each video to get the high-quality subset with `video_cut.json`, add the background annotation at `background_masks.zip`, and most importantly, scene graph annotation from `sailvos3d.json`.

## Step 2: Process the Data

1. Run the script by executing the following command:
   ```
   python rearrange_file.py
   ```
   This script processes the data from the `sailvos3d/` directory and organizes it into the `psg4d_gta/` directory based on the information provided in the `video_cut.json` file.


2. Run the script by executing the following command:
   ```
   python process_data.py
   ```
   The script will process the videos and frames, copying the relevant files to the `psg4d_gta/` directory.

## Step 3: Merge masks
Check that the directory structure and file naming conventions match the expected format:
   ```
   ├── background_masks/
   │   ├── ah_3b_mcs_5/
   │   │   ├── 000000.bmp
   │   │   ├── 000001.bmp
   │   │   └── ...
   │   └── ...
   psg4d_gta/
   ├── camera/
   │   ├── ah_3b_mcs_5/
   │   │   ├── 000000.yaml
   │   │   ├── 000001.yaml
   │   │   └── ...
   │   └── ...
   ├── depth/
   │   ├── ah_3b_mcs_5/
   │   │   ├── 000000.npy
   │   │   ├── 000001.npy
   │   │   └── ...
   │   └── ...
   ├── images/
   │   ├── ah_3b_mcs_5/
   │   │   ├── 000000.bmp
   │   │   ├── 000001.bmp
   │   │   └── ...
   │   └── ...
   ├── rage_matrices/
   │   ├── ah_3b_mcs_5/
   │   │   ├── 000000.npz
   │   │   ├── 000001.npz
   │   │   └── ...
   │   └── ...
   └── visible/
       ├── ah_3b_mcs_5/
       │   ├── 000000.npy
       │   ├── 000001.npy
       │   └── ...
       └── ...
   ```
And run `merge_masks.py`. Then you will see a folder.
```
   psg4d_gta/
   ├── masks/
   │   ├── ah_3b_mcs_5/
   │   │   ├── 000000.yaml
   │   │   ├── 000001.yaml
   │   │   └── ...
   │   └── ...
```
