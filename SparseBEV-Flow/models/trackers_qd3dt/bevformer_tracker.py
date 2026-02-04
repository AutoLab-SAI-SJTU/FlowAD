from copy import deepcopy
import torch
import numpy as np
from .tracklet import Tracklet
from .redundancy import RedundancyModule
from scipy.optimize import linear_sum_assignment
from .frame_data import FrameData
from .update_info_data import UpdateInfoData
from .bbox import BBox, Validity
from .association import associate_dets_to_tracks
import pdb,os


class MOTModel(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.trackers = list()  # tracker for each single tracklet
        self.frame_count = 0  # record for the frames
        self.count = 0  # record the obj number to assign ids
        self.time_stamp = None  # the previous time stamp
        self.redundancy = RedundancyModule(configs)  # module for no detection cases

        non_key_redundancy_config = deepcopy(configs)
        non_key_redundancy_config['redundancy']['det_score_threshold'].update({'giou': 0.1, 'iou': 0.1, 'euler': 0.1})
        non_key_redundancy_config['redundancy']['det_dist_threshold'].update({'giou': -0.5, 'iou': 0.1, 'euler': 4})
        non_key_redundancy_config['redundancy'] = {
            'mode': 'mm',
            'det_score_threshold': non_key_redundancy_config['redundancy']['det_score_threshold'],
            'det_dist_threshold': non_key_redundancy_config['redundancy']['det_dist_threshold']
        }
        self.non_key_redundancy = RedundancyModule(non_key_redundancy_config)

        self.configs = configs
        self.match_type = configs['running']['match_type']
        self.score_threshold = configs['running']['score_threshold']
        self.asso = configs['running']['asso']
        self.asso_thres = configs['running']['asso_thres'][self.asso]
        self.motion_model = configs['running']['motion_model']

        self.max_age = configs['running']['max_age_since_update']
        self.min_hits = configs['running']['min_hits_to_birth']

    def reset(self):
        self.trackers = list()  # tracker for each single tracklet
        self.frame_count = 0  # record for the frames
        self.count = 0  # record the obj number to assign ids
        self.time_stamp = None  # the previous time stamp
        self.redundancy = RedundancyModule(self.configs)  # module for no detection cases

        non_key_redundancy_config = deepcopy(self.configs)
        non_key_redundancy_config['redundancy']['det_score_threshold'].update({'giou': 0.1, 'iou': 0.1, 'euler': 0.1})
        non_key_redundancy_config['redundancy']['det_dist_threshold'].update({'giou': -0.5, 'iou': 0.1, 'euler': 4})
        non_key_redundancy_config['redundancy'] = {
            'mode': 'mm',
            'det_score_threshold': non_key_redundancy_config['redundancy']['det_score_threshold'],
            'det_dist_threshold': non_key_redundancy_config['redundancy']['det_dist_threshold']
        }
        self.non_key_redundancy = RedundancyModule(non_key_redundancy_config)

    @property
    def has_velo(self):
        return not (self.motion_model == 'kf' or self.motion_model == 'fbkf' or self.motion_model == 'ma')

    def frame_mot(self, origin_data, timestamp=None):
        """ For each frame input, generate the latest mot results
        Args:
            input_data (FrameData): input data, including detection bboxes and ego information
        Returns:
            tracks on this frame: [(bbox0, id0), (bbox1, id1), ...]
        """
        # print(origin_data)
        origin_data_copy = deepcopy(origin_data)
        # raw_data = deepcopy(origin_data)
        raw_data = origin_data
        self.frame_count += 1
        input_data = FrameData(raw_data, time_stamp=self.frame_count if timestamp is None else timestamp)
        idxes = -torch.ones_like(raw_data['labels_3d'], dtype=torch.long, device=raw_data['labels_3d'].device)
        classes = deepcopy(raw_data['labels_3d'])

        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp
            # self.time_stamp = self.frame_count

        # if not input_data.aux_info['is_key_frame']:
        #     result = self.non_key_frame_mot(input_data)
        #     return result

        if 'kf' in self.motion_model:
            matched, unmatched_dets, unmatched_trks = self.forward_step_trk(input_data)

        time_lag = input_data.time_stamp - self.time_stamp
        # update the matched tracks
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0]
                        idxes[d] = trk.id
                        classes[d] = torch.topk(trk.class_count, k=1)[1][0]
                        break
                if self.has_velo:
                    aux_info = {
                        'velo': list(input_data.aux_info['velos'][d]),
                        'is_key_frame': input_data.aux_info['is_key_frame']}
                else:
                    aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego,
                                             frame_index=self.frame_count, pc=input_data.pc,
                                             dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
            else:
                result_bbox, update_mode, aux_info = self.redundancy.infer(trk, input_data, time_lag)
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=update_mode, bbox=result_bbox,
                                             ego=input_data.ego, frame_index=self.frame_count,
                                             pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)

        # create new tracks for unmatched detections
        for index in unmatched_dets:
            if self.has_velo:
                aux_info = {
                    'velo': list(input_data.aux_info['velos'][index]),
                    'is_key_frame': input_data.aux_info['is_key_frame']}
            else:
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
            idxes[index] = self.count
            track = Tracklet(self.configs, self.count, input_data.dets[index], input_data.det_types[index],
                                      self.frame_count, aux_info=aux_info, time_stamp=input_data.time_stamp)
            self.trackers.append(track)
            self.count += 1

        # remove dead tracks
        track_num = len(self.trackers)
        for index, trk in enumerate(reversed(self.trackers)):
            if trk.death(self.frame_count):
                self.trackers.pop(track_num - 1 - index)

        # output the results
        # result = list()
        # for trk in self.trackers:
        #     state_string = trk.state_string(self.frame_count)
        #     result.append((trk.get_state(), trk.id, state_string, trk.det_type))

        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp)

        # return result
        index_list = []
        for index, idx in enumerate(idxes):
            if idx >= 0:
                index_list.append(index)
        # origin_data['labels_3d'] = classes
        origin_data.pop('query_feats')
        origin_data.pop('img_feats')
        origin_data.pop('bev_feats')
        for k, v in origin_data.items():
            if k == 'boxes_3d':
                origin_data[k] = v.to('cpu')[index_list]
                # print('\nhhh', origin_data[k], '\n', origin_data_copy[k][index_list])
                continue
            origin_data[k] = v.cpu()[index_list]
        origin_data['track_ids'] = idxes[index_list]

        # print('\n', len(origin_data['track_ids']))
        # print(origin_data['track_ids'])
        # print(origin_data['labels_3d'])
        return origin_data

    def forward_step_trk(self, input_data: FrameData):
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets) if det.s >= self.score_threshold]
        # print('\n', len(det_indexes), len(self.trackers))
        dets = [dets[i] for i in det_indexes]

        # prediction and association
        trk_preds = list()
        # print(self.frame_count, 'before')
        for trk in self.trackers:
            trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame']))
        # print(self.frame_count, 'after')

        # for m-distance association
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers]

        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds,
                                                                           self.match_type, self.asso, self.asso_thres,
                                                                           trk_innovation_matrix)

        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        return matched, unmatched_dets, unmatched_trks

    def non_key_forward_step_trk(self, input_data: FrameData):
        """ tracking on non-key frames (for nuScenes)
        """
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets) if det.s >= 0.5]
        dets = [dets[i] for i in det_indexes]

        # prediction and association
        trk_preds = list()
        for trk in self.trackers:
            trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame']))

        # for m-distance association
        trk_innovation_matrix = None
        if self.asso == 'm_dis':
            trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers]

        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds,
                                                                           self.match_type, self.asso, self.asso_thres,
                                                                           trk_innovation_matrix)

        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]
        return matched, unmatched_dets, unmatched_trks

    def non_key_frame_mot(self, input_data: FrameData):
        """ tracking on non-key frames (for nuScenes)
        """
        self.frame_count += 1
        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = input_data.time_stamp

        if 'kf' in self.motion_model:
            matched, unmatched_dets, unmatched_trks = self.non_key_forward_step_trk(input_data)
        time_lag = input_data.time_stamp - self.time_stamp

        redundancy_bboxes, update_modes = self.non_key_redundancy.bipartite_infer(input_data, self.trackers)
        # update the matched tracks
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0]
                        break
                if self.has_velo:
                    aux_info = {
                        'velo': list(input_data.aux_info['velos'][d]),
                        'is_key_frame': input_data.aux_info['is_key_frame']}
                else:
                    aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=1, bbox=input_data.dets[d], ego=input_data.ego,
                                             frame_index=self.frame_count, pc=input_data.pc,
                                             dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)
            else:
                aux_info = {'is_key_frame': input_data.aux_info['is_key_frame']}
                update_info = UpdateInfoData(mode=update_modes[t], bbox=redundancy_bboxes[t],
                                             ego=input_data.ego, frame_index=self.frame_count,
                                             pc=input_data.pc, dets=input_data.dets, aux_info=aux_info)
                trk.update(update_info)

        # output the results
        result = list()
        for trk in self.trackers:
            state_string = trk.state_string(self.frame_count)
            result.append((trk.get_state(), trk.id, state_string, trk.det_type))

        # wrap up and update the information about the mot trackers
        self.time_stamp = input_data.time_stamp
        for trk in self.trackers:
            trk.sync_time_stamp(self.time_stamp)

        return result