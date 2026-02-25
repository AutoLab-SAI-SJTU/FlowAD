bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2_flow_attn_wm_mlfuse_ego_pe_spatial.py \
    /path/to/your/checkpoint.pth \
    2 \
    --deterministic \
    --eval bbox
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl