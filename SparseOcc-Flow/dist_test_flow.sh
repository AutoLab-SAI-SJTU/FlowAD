#!/usr/bin/env bash

# mount | grep s3_common_dataset


# source /home/work/minianaconda3/etc/profile.d/conda.sh
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate sparsebev

# pip list

PROJECT_FILE="$(dirname "$0")"

cd $PROJECT_FILE

T=`date +%m%d%H%M`    

export PYTHONPATH=`pwd`:$PYTHONPATH

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_8f_flow.py"
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_8f_test.py"
CKPT='./ckpts/sparseocc_r50_nuimg_704x256_8f_60e_v1.1.pth'

# export LLVM_CONFIG=/usr/bin/llvm-config-7
# export NUPLAN_EXP_ROOT="."

GPUS=8
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
NNODES=`expr $GPUS / $GPUS_PER_NODE`

MASTER_PORT=${MASTER_PORT:-11333}
MASTER_ADDR=${MASTER_ADDR:-"127.10.10.1"}
RANK=${RANK:-0}

WORK_DIR=$PROJECT_FILE"/log_wordirs"

# WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# # Intermediate files and logs will be saved to UniAD/projects/work_dirs/


# CUDA_LAUNCH_BLOCKING=1
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
# CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$PROJECT_FILE:$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $PROJECT_FILE"/val.py" \
    --config $CFG \
    --weights $CKPT \
    # --launcher pytorch ${@:4} \
    # --eval bbox \
    # --show-dir ${WORK_DIR} \
    # --deterministic \
    # --tmpdir .
    2>&1 | tee ${WORK_DIR}logs/eval.$T
