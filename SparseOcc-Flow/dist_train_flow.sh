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

GPUS=8
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
NNODES=`expr $GPUS / $GPUS_PER_NODE`

MASTER_PORT=${MASTER_PORT:-18117}
MASTER_ADDR=${MASTER_ADDR:-"127.10.10.2"}
RANK=${RANK:-0}

WORK_DIR=$PROJECT_FILE"/log_wordirs"

# WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# # Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

# CUDA_LAUNCH_BLOCKING=1 \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
PYTHONPATH=$PROJECT_FILE:$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $PROJECT_FILE"/train.py" \
    --config $CFG \
    # --launcher pytorch ${@:4} \
    # --eval bbox \
    # --show-dir ${WORK_DIR} \
    # --deterministic \
    # --tmpdir .
    2>&1 | tee ${WORK_DIR}logs/train.$T
