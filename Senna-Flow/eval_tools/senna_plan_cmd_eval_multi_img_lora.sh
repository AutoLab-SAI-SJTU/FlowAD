export PYTHONPATH="./Senna:$PYTHONPATH"
export HF_HOME="./my_hf_cache"
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=1 python eval_tools/senna_plan_cmd_eval_multi_img_lora.py
