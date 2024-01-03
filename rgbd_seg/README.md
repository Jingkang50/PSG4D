# Segmentation with RGB-D Input

## Get Started
To setup the environment, we use `conda` to manage our dependencies.

Our developers use `CUDA 10.1` to do experiments.

You can specify the appropriate `cudatoolkit` version to install on your machine in the `environment.yml` file, and then run the following to create the `conda` environment:
```bash
conda env create -f environment.yml
conda activate openpvsg
```
You shall manually install the following dependencies.
```bash
# Install mmcv
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
conda install -c conda-forge pycocotools
pip install mmdet==2.25.0

# already within environment.yml
pip install timm
python -m pip install scipy
pip install git+https://github.com/cocodataset/panopticapi.git

# for unitrack
pip install imageio==2.6.1
pip install lap==0.4.0
pip install cython_bbox==0.1.3

# for vps
pip install seaborn
pip install ftfy
pip install regex

# If you're using wandb for logging
pip install wandb
wandb login
```

Download the [pretrained models](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) for tracking if you are interested in IPS+Tracking solution.


## Training and Testing
**IPS+Tracking & Relation Modeling**
```bash
# Train IPS
sh scripts/train/train_ips.sh
# Tracking and save query features
sh scripts/utils/prepare_qf_ips.sh
# Prepare for relation modeling
sh scripts/utils/prepare_rel_set.sh
# Train relation models
sh scripts/train/train_relation.sh
# Test
sh scripts/test/test_relation_full.sh
```