#!/usr/bin/env bash

PROJECT_FILE="."

cd $PROJECT_FILE

T=`date +%m%d%H%M`    

export PYTHONPATH=`pwd`:$PYTHONPATH

# CONFIG="projects/configs/sparsedrive_small_stage2_flow_attn_test.py"
# CHECKPOINT="/data3/guomingzhe/SparseDrive/work_dirs/sparsedrive_small_stage2_flow_attn/iter_11715.pth"

# CONFIG="projects/configs/sparsedrive_small_stage1.py"
# CHECKPOINT="/data3/guomingzhe/SparseDrive/ckpt/sparsedrive_stage1.pth"

# CONFIG="projects/configs/sparsedrive_small_stage1_flow_attn.py"
# CHECKPOINT="/data3/guomingzhe/SparseDrive/work_dirs/sparsedrive_small_stage1_flow_attn/iter_58600.pth"

# CONFIG="projects/configs/sparsedrive_small_stage2_flow_attn_test.py"
# CHECKPOINT="/data3/guomingzhe/SparseDrive/work_dirs/sparsedrive_small_stage2_flow_attn/latest.pth"

CONFIG="projects/configs/sparsedrive_small_stage1_flow_attn_wm_mlfuse_ego_pe_spatial.py"
# CHECKPOINT="/data3/guomingzhe/SparseDrive/work_dirs/sparsedrive_small_stage2_flow_attn_wm_mlfuse2/iter_39050.pth"
CHECKPOINT="./pretrain/iter_45695_final_s1.pth"

# CONFIG="projects/configs/sparsedrive_small_stage1_flow_attn_wm_mlfuse.py"
# CHECKPOINT="/data3/guomingzhe/SparseDrive/work_dirs/sparsedrive_small_stage1_flow_attn_wm_mlfuse/iter_58600.pth"

echo $CONFIG
echo $CHECKPOINT

GPUS=8
PORT=${PORT:-26651}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES=1 \
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $PROJECT_FILE"/tools/test.py" $CONFIG $CHECKPOINT --deterministic \
    --eval bbox --launcher pytorch ${@:4}