import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mmcv

def command2hot(command,max_dim=6):
    if command < 0:
        command = 4
    command -= 1
    cmd_one_hot = np.zeros(max_dim)
    cmd_one_hot[command] = 1
    return cmd_one_hot

def get_ego_trajs(data_infos,idx,sample_rate,past_frames,future_frames, use_cmd=True):
    adj_idx_list = range(idx-past_frames*sample_rate,idx+(future_frames+1)*sample_rate,sample_rate)
    cur_frame = data_infos[idx]
    full_adj_track = np.zeros((past_frames+future_frames+1,2))
    full_adj_adj_mask = np.zeros(past_frames+future_frames+1)
    world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
    for j in range(len(adj_idx_list)):
        adj_idx = adj_idx_list[j]
        if adj_idx <0 or adj_idx>=len(data_infos):
            break
        adj_frame = data_infos[adj_idx]
        if adj_frame['folder'] != cur_frame ['folder']:
            break
        world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
        adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
        xy = adj2cur_lidar[0:2,3]
        full_adj_track[j,0:2] = xy
        full_adj_adj_mask[j] = 1
    offset_track = full_adj_track[1:] - full_adj_track[:-1]
    for j in range(past_frames-1,-1,-1):
        if full_adj_adj_mask[j] == 0:
            offset_track[j] = offset_track[j+1]
    for j in range(past_frames,past_frames+future_frames,1):

        if full_adj_adj_mask[j+1] == 0 :
            offset_track[j] = 0
    if use_cmd:
        command = command2hot(cur_frame['command_near'])
    else:
        command = np.array([0, ])
    offset_track = offset_track.astype(np.float32)
    return offset_track[:past_frames].copy(), offset_track[past_frames:].copy(), full_adj_adj_mask[-future_frames:].copy(), command
    

K = 6
command_type = 6

fp = 'data/infos/b2d_infos_val.pkl'
data = mmcv.load(fp)
# data_infos = list(sorted(data, key=lambda e: e["timestamp"]))
data_infos = data
navi_trajs = [[] for i in range(command_type)]
for idx in tqdm(range(len(data_infos))):
    info = data_infos[idx]
    # plan_traj = info['gt_ego_fut_trajs'].cumsum(axis=-2)
    # plan_mask = info['gt_ego_fut_masks']
    # cmd = info['gt_ego_fut_cmd'].astype(np.int32)

    (
            ego_his_trajs, 
            plan_traj, 
            plan_mask, 
            cmd
        ) = get_ego_trajs(data_infos,idx,1,2,6)

    cmd = cmd.argmax(axis=-1)
    if not plan_mask.sum() == 6:
        continue
    # print(navi_trajs)
    # print(cmd)
    navi_trajs[cmd].append(plan_traj)

clusters = []
for trajs in navi_trajs:
    trajs = np.concatenate(trajs, axis=0).reshape(-1, 12)
    cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_
    cluster = cluster.reshape(-1, 6, 2)
    clusters.append(cluster)
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1])
plt.savefig(f'vis/kmeans/plan_{K}', bbox_inches='tight')
plt.close()

clusters = np.stack(clusters, axis=0)
np.save(f'data/kmeans/kmeans_plan_{K}.npy', clusters)