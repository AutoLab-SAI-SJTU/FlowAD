# SparseDrive-Flow

This project extends [SparseDrive](https://github.com/swc-17/SparseDrive) with our flow-based framework for enhanced end-to-end autonomous driving.

> **Note**: For paper information and citation, please refer to the [main README](../README.md).

## Flow Method Extension

This repository adds flow-based feature enhancement to the original SparseDrive model for improved motion prediction and planning.

### Flow-specific Files

- **Models**: `projects/mmdet3d_plugin/models/sparsedrive_flow*.py`
- **Configs**: `projects/configs/sparsedrive_*_flow*.py`
- **Training Scripts**: `scripts/train_flowad*.sh`

<!-- ### Flow Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `flow_patches` | Patch sizes for multi-scale flow extraction | `[8, 4, 2, 1]` |
| `flow_ids` | Feature level indices for flow processing | `[0, 1, 2, 3]` | -->

### Recommended Flow Configs

For best results, use the following configs:

**ResNet-50 Backbone:**
- Stage 1: `projects/configs/sparsedrive_small_stage1_flow_attn_wm_mlfuse_ego_pe_spatial.py`
- Stage 2: `projects/configs/sparsedrive_small_stage2_flow_attn_wm_mlfuse_ego_pe_spatial.py`

**ResNet-101 Backbone:**
- Stage 1: `projects/configs/sparsedrive_small_stage1_flow_attn_wm_mlfuse_ego_pe_spatial_r101.py`
- Stage 2: `projects/configs/sparsedrive_small_stage2_flow_attn_wm_mlfuse_ego_pe_spatial_r101.py`


## Quick Start


### Set up a new virtual environment
```bash
conda create -n sparsedrive python=3.8 -y
conda activate sparsedrive
```

### Install dependency packpages
```bash
sparsedrive_path="path/to/sparsedrive"
cd ${sparsedrive_path}
pip3 install --upgrade pip
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install -r requirement.txt
```

### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and CAN bus expansion, put CAN bus expansion in /path/to/nuscenes, create symbolic links.
```bash
cd ${sparsedrive_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

You should download these files([nuscenes_infos_train_ego.pkl](https://github.com/AutoLab-SAI-SJTU/FlowAD/releases/download/v1.0.0/nuscenes_infos_train_ego.pkl) and [nuscenes_infos_val_ego.pkl](https://github.com/AutoLab-SAI-SJTU/FlowAD/releases/download/v1.0.0/nuscenes_infos_val_ego.pkl)) and put them to the path data/infos for training.


Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter, with a roi_size of (30, 60) as default, if you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.
```bash
sh scripts/create_data.sh
```

### Generate anchors by K-means
Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```bash
sh scripts/kmeans.sh
```


### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Commence training and testing

Before training, you should modify `num_gpus` in the config files to match your actual GPU count. The total batch size is fixed, and per-GPU batch size will be automatically computed:

```
# in projects/configs/sparsedrive_small_stage1_flow_attn_wm_mlfuse_ego_pe_spatial.py
num_gpus = 2              # modify to match your actual GPU count
total_batch_size = 32     # fixed, do not change
batch_size = total_batch_size // num_gpus  # auto-computed

# in projects/configs/sparsedrive_small_stage2_flow_attn_wm_mlfuse_ego_pe_spatial.py
num_gpus = 2              # modify to match your actual GPU count
total_batch_size = 64     # fixed, do not change
batch_size = total_batch_size // num_gpus  # auto-computed
```

Also make sure the GPU count in `scripts/train_flowad.sh` matches `num_gpus` in the config.

```bash
# train
sh scripts/train_flowad.sh

# test
sh scripts/test_flowad.sh
```

### Visualization
```
sh scripts/visualize.sh
```



## Acknowledgement

This project is based on [SparseDrive](https://github.com/swc-17/SparseDrive). Thanks to the original authors for their excellent work.

Additional references:
- [Sparse4D](https://github.com/HorizonRobotics/Sparse4D)
- [UniAD](https://github.com/OpenDriveLab/UniAD) 
- [VAD](https://github.com/hustvl/VAD)
- [StreamPETR](https://github.com/exiawsh/StreamPETR)
- [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
