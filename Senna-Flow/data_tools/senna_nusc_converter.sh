export http_proxy=http://172.19.56.199:3128

export https_proxy=http://172.19.56.199:3128

export no_proxy=bcebos.com

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=./Senna:$PYTHONPATH

echo "Start generating dataset..."

python \
    data_tools/senna_nusc_data_converter.py \
    nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./output \
    --extra-tag senna_nusc \
    --version v1.0 \
    --canbus ./data/nuscenes
