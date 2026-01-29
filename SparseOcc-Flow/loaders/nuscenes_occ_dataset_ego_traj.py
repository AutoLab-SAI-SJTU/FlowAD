import os
import mmcv
import glob
import torch
import numpy as np
from tqdm import tqdm
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import DataLoader
from models.utils import sparse2dense
from .ray_metrics import main_rayiou, main_raypq
from .ego_pose_dataset import EgoPoseDataset
from configs.r50_nuimg_704x256_8f import occ_class_names as occ3d_class_names
from configs.r50_nuimg_704x256_8f_openocc import occ_class_names as openocc_class_names

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

@DATASETS.register_module()
class NuSceneOccEgoTraj(NuScenesDataset):    
    def __init__(self, occ_gt_root, *args, **kwargs):
        super().__init__(filter_empty_gt=False, *args, **kwargs)
        self.occ_gt_root = occ_gt_root
        self.data_infos = self.load_annotations(self.ann_file)

        self.token2scene = {}
        for gt_path in glob.glob(os.path.join(self.occ_gt_root, '*/*/*.npz')):
            token = gt_path.split('/')[-2]
            scene_name = gt_path.split('/')[-3]
            self.token2scene[token] = scene_name

        for i in range(len(self.data_infos)):
            scene_name = self.token2scene[self.data_infos[i]['token']]
            self.data_infos[i]['scene_name'] = scene_name

    def get_ego_poses(self, index):
        info = self.data_infos[index]
        can_bus = np.copy(info['can_bus'])
        rotation = Quaternion(info['ego2global_rotation'])
        translation = info['ego2global_translation']
        # print(can_bus, translation, rotation, rotation.rotation_matrix)
        # can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle


        # sd_rec = self.nusc.get('sample', info['token'])
        # lidar_top_data_start = self.nusc.get('sample_data', sd_rec['data']['LIDAR_TOP'])
        # ego_pose_start = self.nusc.get('ego_pose', lidar_top_data_start['ego_pose_token'])

        # sdc_fut_traj = []
        # for _ in range(2):
        #     next_annotation_token = sd_rec['next']
        #     if next_annotation_token=='':
        #         break
        #     sd_rec = self.nusc.get('sample', next_annotation_token)
        #     lidar_top_data_next = self.nusc.get('sample_data', sd_rec['data']['LIDAR_TOP'])
        #     ego_pose_next = self.nusc.get('ego_pose', lidar_top_data_next['ego_pose_token'])
        #     sdc_fut_traj.append(ego_pose_next['translation'][:2])  # global xy pos of sdc at future step i
        # print(sdc_fut_traj, ego_pose_start['translation'], ego_pose_start['rotation'])
        # print(info['ego2global_translation'], info['ego2global_rotation'], )
        
        # # sdc_fut_traj_all = np.zeros((1, 6, 2))
        # # sdc_fut_traj_valid_mask_all = np.zeros((1, 6, 2))
        # n_valid_timestep = len(sdc_fut_traj)
        # if n_valid_timestep>0:
        #     sdc_fut_traj = np.stack(sdc_fut_traj, axis=0)  #(t,2)
        #     sdc_fut_traj = convert_global_coords_to_local(
        #         coordinates=sdc_fut_traj,
        #         translation=ego_pose_start['translation'],
        #         rotation=ego_pose_start['rotation'],
        #     )
        #     print(sdc_fut_traj)
        # raise EOFError

        return can_bus, [info['ego2global_translation'], info['ego2global_rotation']]

    def collect_sweeps(self, index, into_past=150, into_future=0):
        ego_poses_prev = []
        all_sweeps_prev = []
        trans_rot_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            ego_poses, trans_rot = self.get_ego_poses(curr_index)
            ego_poses_prev.append(ego_poses)
            trans_rot_prev.append(trans_rot)
            curr_index = curr_index - 1
        
        ego_poses_next = []
        all_sweeps_next = []
        trans_rot_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            ego_poses, trans_rot = self.get_ego_poses(curr_index)
            ego_poses_next.append(ego_poses)
            trans_rot_next.append(trans_rot)
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next, ego_poses_prev, ego_poses_next, trans_rot_prev, trans_rot_next

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next, ego_poses_prev, ego_poses_next, trans_rot_prev, trans_rot_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix

        ego_poses, trans_rots = self.get_ego_poses(index)

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
            ego_pose_expand={'prev': ego_poses_prev, 'next': ego_poses_next},
            ego_pose=[ego_poses],
            trans_rot_expand={'prev': trans_rot_prev, 'next': trans_rot_next},
            trans_rot=[trans_rots],
        )

        ego2lidar = transform_matrix(lidar2ego_translation, Quaternion(lidar2ego_rotation), inverse=True)
        input_dict['ego2lidar'] = [ego2lidar for _ in range(6)]
        input_dict['occ_path'] = os.path.join(self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts, occ_preds, inst_gts, inst_preds, lidar_origins = [], [], [], [], []
        print('\nStarting Evaluation...')

        sample_tokens = [info['token'] for info in self.data_infos]

        for batch in DataLoader(EgoPoseDataset(self.data_infos), num_workers=8):
            token = batch[0][0]
            output_origin = batch[1]
            
            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            occ_path = os.path.join(self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)
            gt_semantics = occ_gt['semantics']

            occ_pred = occ_results[data_id]
            sem_pred = torch.from_numpy(occ_pred['sem_pred'])  # [B, N]
            occ_loc = torch.from_numpy(occ_pred['occ_loc'].astype(np.int64))  # [B, N, 3]
            
            data_type = self.occ_gt_root.split('/')[-1]
            if data_type == 'occ3d' or data_type == 'occ3d_panoptic':
                occ_class_names = occ3d_class_names
            elif data_type == 'openocc_v2':
                occ_class_names = openocc_class_names
            else:
                raise ValueError
            free_id = len(occ_class_names) - 1
            
            occ_size = list(gt_semantics.shape)
            sem_pred, _ = sparse2dense(occ_loc, sem_pred, dense_shape=occ_size, empty_value=free_id)
            sem_pred = sem_pred.squeeze(0).numpy()

            if 'pano_inst' in occ_pred.keys():
                pano_inst = torch.from_numpy(occ_pred['pano_inst'])
                pano_sem = torch.from_numpy(occ_pred['pano_sem'])

                pano_inst, _ = sparse2dense(occ_loc, pano_inst, dense_shape=occ_size, empty_value=0)
                pano_sem, _ = sparse2dense(occ_loc, pano_sem, dense_shape=occ_size, empty_value=free_id)
                pano_inst = pano_inst.squeeze(0).numpy()
                pano_sem = pano_sem.squeeze(0).numpy()
                sem_pred = pano_sem

                gt_instances = occ_gt['instances']
                inst_gts.append(gt_instances)
                inst_preds.append(pano_inst)

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(sem_pred)
        
        if len(inst_preds) > 0:
            results = main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins, occ_class_names=occ_class_names)
            results.update(main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names))
            return results
        else:
            return main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names)

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        
        print('\nFinished.')
