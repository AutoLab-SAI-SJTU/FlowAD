#!/usr/bin/env bash

# mount | grep s3_common_dataset


# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate torch2

# pip list

PROJECT_FILE="$(dirname "$0")"

cd $PROJECT_FILE

T=`date +%m%d%H%M`    

export PYTHONPATH=`pwd`:$PYTHONPATH

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256.py"
# CKPT="."

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_nocliploss.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip2.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip3.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip4.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip5.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_stage.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_stage2.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl.py"
# CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_nolanggrad.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_nodetach.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_simpredict.py"
CKPT='.


# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_vl2.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl2.py"
# CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_re.py"
CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_vislevel.py"
# CKPT='.


# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl4.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl5_repeat.py"
# CKPT='.


# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_template.py"
# CKPT='.


# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each.py"
# CKPT='.


# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip3.py"
# CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_simpredict_2dmap.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_simpredict_2dmap_nodnloss.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_detach.py"
CKPT='.
# CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_debug.py"
CKPT='.
# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_re.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_re.py"
# CKPT='.


# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d2.py"
# CKPT='.


# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance.py"
# CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_sparse.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one2.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one3.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_cluster.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_neck.py"
CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_contrasive.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_cluster2.py"
# CKPT='.

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_cluster3.py"
# CKPT='.


# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_simpredict.py"
# CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_decodetgt.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_decodetgt2.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_contrasive_fp.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_prior.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one4.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_decodetgt2_1.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_decodetgt2_2.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_decodetgt2_8.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt2.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat5.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat11.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_decodetgt2_samheat2.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat18.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat20.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat22.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_decodetgt2_samheat9.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat27.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat28.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_onlyenhance4.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_decodetgt2_33.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat30.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_cycer.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_cycer667.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat131.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_joint.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat132.py"
CKPT='.

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery3.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery3/epoch_4.pth'

# 0.34
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery5.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery5/epoch_4.pth'

# 0.3487 0.4622
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance/epoch_4.pth'

# 0.3510 0.4692
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance2.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance2/epoch_4.pth'

# 0.2989 0.4153
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance3.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance3/epoch_4.pth'

# 0.3434 0.4599
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance4.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis_noenhance/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance4/epoch_4.pth'

# 0.3103 0.4311 8frame -> 1frame
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance5.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance5/epoch_4.pth'

# 0.3365 0.4493
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance6.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance6/epoch_4.pth'

# 0.3434 0.4543
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis_mae/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae/epoch_4.pth'

# 0.2943
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance9.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance9/epoch_4.pth'

# 0.3472 0.4669
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance10.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance10/epoch_4.pth'

# 0.3523 0.4565
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance11.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance11/epoch_4.pth'

# 0.2849 0.4065
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae2.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis_mae/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae2/epoch_4.pth'

# 0.3546 0.4681
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance13.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance13/epoch_4.pth'

# 0.2742 0.3946
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae3.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis_mae/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae3/epoch_4.pth'

# 0.3312 0.4467
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae4.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis_mae/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae4/epoch_4.pth'

# 0.3395 0.4573 
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae5.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis_mae/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae5/epoch_4.pth'

# 0.3337 0.4446
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae6.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis_mae/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae6/epoch_4.pth'

# 0.3400  0.4528
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae7.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis_mae/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae7/epoch_4.pth'

# 0.3469 0.4617
CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae8.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis_mae/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae8/epoch_4.pth'

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae9.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis_mae/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae9/epoch_24.pth'

# CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae92.py"
# CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis_mae/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance_mae92/epoch_4.pth'

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance13.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance13/epoch_20.pth'
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance13_re/iter_400.pth'
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance13_re/iter_400.pth'

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256.py"
CKPT="./outputs/SparseBEV/r50_nuimg_704x256/epoch_4.pth"

CFG=$PROJECT_FILE"/configs/r101_nuimg_1408x512.py"
CKPT="."

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance13_fine.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance13_fine/epoch_6.pth'

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance13.py"
CKPT='./outputs/SparseBEV_CLIP_2d_enhance_one_clsvis/r50_nuimg_704x256_clip_serial_each_vl_2d_enhance_one_clsvis_neck_decodetgt_samheat_2dquery_noenhance13_re_rebuttal/iter_350.pth'

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_flow.py"
CKPT='./outputs/SparseBEV/r50_nuimg_704x256_flow/epoch_24.pth'

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_flow_attn.py"
CKPT='./outputs/SparseBEV/r50_nuimg_704x256_flow_attn/epoch_24.pth'

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_flow_attn2.py"
CKPT='./outputs/SparseBEV/r50_nuimg_704x256_flow_attn2/epoch_24.pth'

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_flow_attn_wm2.py"
CKPT='.-Flow/outputs/SparseBEV/r50_nuimg_704x256_flow_attn_wm2/epoch_6.pth'

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_flow_attn_sample_test.py"
CKPT='.-Flow/outputs/SparseBEV/r50_nuimg_704x256_flow_attn_sample/epoch_6.pth'

CFG=$PROJECT_FILE"/configs/r50_nuimg_704x256_flow_attn_wm_mlfuse_test.py"
CKPT='.-Flow/outputs/SparseBEV/r50_nuimg_704x256_flow_attn_wm_mlfuse/epoch_23.pth'

# export LLVM_CONFIG=/usr/bin/llvm-config-7
# export NUPLAN_EXP_ROOT="."

GPUS=1
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
NNODES=`expr $GPUS / $GPUS_PER_NODE`

MASTER_PORT=${MASTER_PORT:-11333}
MASTER_ADDR=${MASTER_ADDR:-"127.10.10.1"}
RANK=${RANK:-0}

WORK_DIR="._wordirs"

# WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# # Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

# CUDA_LAUNCH_BLOCKING=1
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
CUDA_VISIBLE_DEVICES=1 \
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
