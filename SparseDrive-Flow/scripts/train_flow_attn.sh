
PROJECT_FILE="/data3/guomingzhe/SparseDrive"

cd $PROJECT_FILE

T=`date +%m%d%H%M`    

export PYTHONPATH=`pwd`:$PYTHONPATH

CONFIG="projects/configs/sparsedrive_small_stage2_flow_attn.py"
GPUS=1
PORT=${PORT:-28651}

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
CUDA_VISIBLE_DEVICES=1 \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $PROJECT_FILE"/tools/train.py" $CONFIG --launcher pytorch ${@:3}
