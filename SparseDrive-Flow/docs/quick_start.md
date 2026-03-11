# Quick Start

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

You should download these [files]() and put them to the path data/infos.

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
