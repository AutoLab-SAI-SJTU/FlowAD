#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28651}

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
