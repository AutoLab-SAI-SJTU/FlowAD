export http_proxy=http://172.19.56.199:3128
export https_proxy=http://172.19.56.199:3128
export no_proxy=bcebos.com

export HF_HOME="./my_hf_cache"
export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=./Senna:$PYTHONPATH

export LD_LIBRARY_PATH=/home/work/miniconda3/envs/senna/lib/python3.8/site-packages/torch/lib/:$LD_LIBRARY_PATH
# export CUDA_VISIBLE_DEVICES=0

DATA="./Senna/infos/senna_nusc_val.json"

python eval_tools/senna_plan_visualization.py \
    --eval-data-path ./Senna/infos/senna_nusc_select_all.json \
    --model-path ./Senna/output-senna-flow \
    --save-path ./Senna/vis_result