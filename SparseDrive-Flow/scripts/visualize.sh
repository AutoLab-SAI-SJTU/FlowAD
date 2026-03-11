export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization/visualize.py \
	projects/configs/sparsedrive_small_stage2_flow_attn_wm_mlfuse_ego_pe_spatial.py \
	--result-path work_dirs/sparsedrive_small_stage2/results.pkl