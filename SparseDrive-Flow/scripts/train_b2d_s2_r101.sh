
PROJECT_FILE="."

cd $PROJECT_FILE

T=`date +%m%d%H%M`    

export PYTHONPATH=`pwd`:$PYTHONPATH

# CONFIG="projects/configs_b2d/sparsedrive_small_b2d_stage2.py"
CONFIG="projects/configs_b2d/sparsedrive_small_flow_b2d_stage2_r101.py"
GPUS=8
PORT=${PORT:-18651}

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $PROJECT_FILE"/tools/train.py" $CONFIG --launcher pytorch ${@:3}
