## stage1
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage1_flow_attn_wm_mlfuse_ego_pe_spatial.py \
   2 \
   --deterministic

## stage2
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage2_flow_attn_wm_mlfuse_ego_pe_spatial.py \
   2 \
   --deterministic