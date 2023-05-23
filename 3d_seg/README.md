# DKNet (Dynamic Kernel Network)

## Installation

### Requirements
* Python 3.7.0
* Pytorch 1.5.0
* CUDA 10.2

### Virtual Environment
```bash
conda create -n dknet python==3.7
source activate dknet
```

### Install DKNet

(1) Install the dependent libraries.
```bash
pip install -r requirements.txt
conda install -c bioconda google-sparsehash
```

(2) Install spconv

We use spconv2.x. Please refer to [spconv](https://github.com/traveller59/spconv) for details.

(3) Compile the external C++ and CUDA ops.
* Install dknet_ops
```bash
cd ./lib/dknet_ops
export CPLUS_INCLUDE_PATH={conda_env_path}/dknet/include:$CPLUS_INCLUDE_PATH
python setup.py build_ext develop
```
{conda_env_path} is the location of the created conda environment, e.g., `/anaconda3/envs`.
Alternative installation guide can be found in [here](https://github.com/hustvl/HAIS).

* Install segmentator

Build example:
```bash
cd ./lib/segmentator

cd csrc && mkdir build && cd build

cmake .. -DCMAKE_C_COMPILER=/mnt/lustre/share/gcc/gcc-5.3.0/bin/gcc -DCMAKE_CXX_COMPILER=/mnt/lustre/share/gcc/gcc-5.3.0/bin/g++ -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCUDA_TOOLKIT_ROOT_DIR=/mnt/lustre/share/cuda-10.2 \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'`

make && make install # after install, please do not delete this folder (as we only create a symbolic link)
```

Further information can be found in [here](https://github.com/Karbo123/segmentator).

## Data Preparation

### Sailvos3D
The 3D point clouds data could be generated following:
```
cd dataset/saivos3d
python generate_3d_data.py
```
Don't forget to change the corresponding input and output path.

### Hoi4D
The 3D point clouds data could be generated following:
```
cd dataset/hoi4d/HOI4D-Instructions/prepare_4Dseg
python prepare_4Dseg_dataset.py
```
Don't forget to change the corresponding input and output path.

## Training
### Sailvos3D
```bash
python -m torch.distributed.launch --nproc_per_node=4 train_sailvos3d.py --config /mnt/lustre/jcen/dknet/config/DKNet_run1_sailvos3d_100e_dis_train.yaml
```
### Hoi4D
```bash
python -m torch.distributed.launch --nproc_per_node=8 train_hoi4d.py --config /mnt/lustre/jcen/dknet/config/DKNet_run1_hoi4d_100e_dis_train_train.yaml
```

## Inference
We need to obtain the 3D panoptic segmentation results for both the train and test sets, and they will be used for the downstram tracking and relation model.
### Sailvos3D
```bash
python -m torch.distributed.launch --nproc_per_node=4 test_sailvos3d.py --config /mnt/lustre/jcen/dknet/config/DKNet_run1_sailvos3d_100e_dis_test_trainset.yaml --pretrain exp/sailvos3d/DKNet/DKNet_run1_sailvos3d_100e_dis_train/DKNet_run1_sailvos3d_100e_dis_train-000000100.pth
```
```bash
python -m torch.distributed.launch --nproc_per_node=4 test_sailvos3d.py --config /mnt/lustre/jcen/dknet/config/DKNet_run1_sailvos3d_100e_dis_test_testset.yaml --pretrain exp/sailvos3d/DKNet/DKNet_run1_sailvos3d_100e_dis_train/DKNet_run1_sailvos3d_100e_dis_train-000000100.pth
```
The results will be stored at `exp/sailvos3d/DKNet/*/result/epoch100_scoret0.1_npointt100/val/segmentation_results`
### Hoi4D
```bash
python -m torch.distributed.launch --nproc_per_node=4 test_hoi4d.py --config /mnt/lustre/jcen/dknet/config/DKNet_run1_hoi4d_100e_dis_test_trainset.yaml --pretrain exp/hoi4d/DKNet/DKNet_run1_hoi4d_100e_dis_train/DKNet_run1_hoi4d_100e_dis_train-000000100.pth
```
```bash
python -m torch.distributed.launch --nproc_per_node=4 test_hoi4d.py --config /mnt/lustre/jcen/dknet/config/DKNet_run1_hoi4d_100e_dis_test_testset.yaml --pretrain exp/hoi4d/DKNet/DKNet_run1_hoi4d_100e_dis_train/DKNet_run1_hoi4d_100e_dis_train-000000100.pth
```
The results will be stored at `exp/hoi4d/DKNet/*/result/epoch100_scoret0.1_npointt100/val/segmentation_results`

## Tracking
```bash
cd ../3d_track
python test_query_tube_dknet.py
```
Don't forget to change the corresponding config files. We provide all config files in `3d_track/configs/unitrack`

## Acknowledgement
This repo is built upon several repos, e.g.,  [spconv](https://github.com/traveller59/spconv), [PointGroup](https://github.com/dvlab-research/PointGroup) and [DyCo3D](https://github.com/aim-uofa/DyCo3D).

