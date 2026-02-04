import numpy as np
import torch
from .kalman_filter import KalmanFilterMotionModel
from .hit_manager import HitManager
from .update_info_data import UpdateInfoData
from .frame_data import FrameData
from .bbox import BBox


class Tracklet:
    def __init__(self, configs, id, bbox: BBox, det_type, frame_index, time_stamp=None, aux_info=None):
        self.id = id
        self.time_stamp = time_stamp
        self.asso = configs['running']['asso']

        self.configs = configs
        self.det_type = det_type
        self.aux_info = aux_info

        # initialize different types of motion model
        self.motion_model_type = configs['running']['motion_model']
        # simple kalman filter
        if self.motion_model_type == 'kf':
            self.motion_model = KalmanFilterMotionModel(
                bbox=bbox, inst_type=self.det_type, time_stamp=time_stamp, covariance=configs['running']['covariance'])

        # life and death management
        self.life_manager = HitManager(configs, frame_index)
        # store the score for the latest bbox
        self.latest_score = bbox.s
        self.query_feat = bbox.query_feat
        self.img_feat = bbox.img_feat
        self.bev_feat = bbox.bev_feat
        self.class_count = torch.zeros(7, dtype=torch.long, device=self.query_feat.device)
        self.class_count[det_type] = self.class_count[det_type]+1

    def predict(self, time_stamp=None, is_key_frame=True):
        """ in the prediction step, the motion model predicts the state of bbox
            the other components have to be synced
            the result is a BBox

            the ussage of time_stamp is optional, only if you use velocities
        """
        result = self.motion_model.get_prediction(time_stamp=time_stamp)
        self.life_manager.predict(is_key_frame=is_key_frame)
        self.latest_score = self.latest_score * 0.01
        result.s = self.latest_score
        result.query_feat = self.query_feat
        result.img_feat = self.img_feat
        result.bev_feat = self.bev_feat
        result.class_count = self.class_count
        return result

    def update(self, update_info: UpdateInfoData):
        """ update the state of the tracklet
        """
        self.latest_score = update_info.bbox.s
        self.query_feat = update_info.bbox.query_feat
        self.img_feat = update_info.bbox.img_feat
        self.bev_feat = update_info.bbox.bev_feat
        is_key_frame = update_info.aux_info['is_key_frame']

        # only the direct association update the motion model
        if update_info.mode == 1 or update_info.mode == 3:
            self.motion_model.update(update_info.bbox, update_info.aux_info)
            self.class_count[update_info.bbox.label] = self.class_count[update_info.bbox.label] + 1
        else:
            pass
        self.life_manager.update(update_info, is_key_frame)
        return

    def get_state(self):
        """ current state of the tracklet
        """
        result = self.motion_model.get_state()
        result.s = self.latest_score
        return result

    def valid_output(self, frame_index):
        return self.life_manager.valid_output(frame_index)

    def death(self, frame_index):
        return self.life_manager.death(frame_index)

    def state_string(self, frame_index):
        """ the string describes how we get the bbox (e.g. by detection or motion model prediction)
        """
        return self.life_manager.state_string(frame_index)

    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return self.motion_model.compute_innovation_matrix()

    def sync_time_stamp(self, time_stamp):
        """ sync the time stamp for motion model
        """
        self.motion_model.sync_time_stamp(time_stamp)
        return
