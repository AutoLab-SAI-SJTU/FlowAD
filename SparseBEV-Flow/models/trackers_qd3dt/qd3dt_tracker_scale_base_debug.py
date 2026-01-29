from copy import deepcopy
import random
import torch
import torch.nn.functional as F
import numpy as np
from mmcv.ops import diff_iou_rotated_3d
from scipy.optimize import linear_sum_assignment as linear_assignment
from .tracklet import Tracklet
from .redundancy import RedundancyModule
from scipy.optimize import linear_sum_assignment
from .frame_data import FrameData
from .update_info_data import UpdateInfoData
from .bbox import BBox, Validity
from .association import associate_dets_to_tracks
from .bbox import bbox_overlaps
from .kalman_filter_qd3dt import KalmanBox3DTracker, LSTM3DTracker
from .motion_model_lstm import get_lstm_model
from .association import all_iou
import pdb, os

# def set_random_seed(seed, deterministic=False):
#     """Set random seed.
#
#     Args:
#         seed (int): Seed to be used.
#         deterministic (bool): Whether to set the deterministic option for
#             CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
#             to True and `torch.backends.cudnn.benchmark` to False.
#             Default: False.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     if deterministic:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
# set_random_seed(0, deterministic=True)




class MOTModel(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.tracker_model_name = 'KF'  # 'KF / LSTM3DTracker'

        if self.tracker_model_name == 'LSTM3DTracker':
            self.tracker_model = LSTM3DTracker
            self.lstm = None
            self.lstm_name = 'VeloLSTM'
            self.lstm_ckpt_name = 'J:/BDD100K/batch128_min10_seq10_dim7_VeloLSTM_nuscenes_100_linear.pth'
        else:
            self.tracker_model = KalmanBox3DTracker

        self.init_score_thr = 0.25  # 0.5
        self.init_track_id = 0
        self.obj_score_thr = 0.6
        self.match_score_thr = [0.55, 0.5]  # [0.3, 0.5]  # 0.5
        self.memo_tracklet_frames = 2  # 10
        self.memo_backdrop_frames = 0
        self.memo_momentum = 0.5
        self.motion_momentum = 0.5
        self.nms_conf_thr = 0.5
        self.nms_backdrop_iou_thr = 0.3
        self.nms_class_iou_thr = 0.7
        self.loc_dim = 7
        # self.init_score_thr = 0.5  # 0.5
        # self.init_track_id = 0
        # self.obj_score_thr = 0.5
        # self.match_score_thr = 0.5
        # self.memo_tracklet_frames = 10
        # self.memo_backdrop_frames = 1
        # self.memo_momentum = 0.8
        # self.motion_momentum = 0.8
        # self.nms_conf_thr = 0.5
        # self.nms_backdrop_iou_thr = 0.3
        # self.nms_class_iou_thr = 0.7
        # self.loc_dim = 7

        self.with_depth_uncertainty = True
        self.with_bbox_iou = True
        self.track_bbox_iou = 'box3d'  # 'box3d / bev / alliou / iou3d'
        self.with_query_feat = True
        self.with_bev_feat = True
        self.with_img_feat = True
        self.match_metric = 'norm_product'  # 'norm_product / cosine / cycle_softmax'
        self.with_depth_ordering = False
        self.depth_match_metric = 'centroid'  # 'motion / centroid / pure_motion'
        self.with_cats = True
        self.bbox_affinity_weight = 0.3 if self.with_query_feat else 1.0  # 0.3
        self.feat_affinity_weight = 1 - self.bbox_affinity_weight if self.with_bbox_iou else 1.0
        self.weight_query_feat = 1.  # 1 / 3
        self.weight_bev_feat = 0.  # 1 / 3
        self.weight_img_feat = 0.  # 1 / 3
        self.match_algo = 'greedy'  # 'greedy / hungarian'

        # self.with_vxy = False
        self.buffer_sizes = [2.0, 1.5, 1.0, 0.5, 0.]  # [0.5, 0.4, 0.3, 0.2, 0.1]
        self.buffer_bbox = False  # attn!!!
        self.with_scales = [False, False]
        self.multi_scale_matching = [False, False]  # including self.with_scales
        self.re_matching = False  # if re-match after self.multi_scale_matching

        self.bev_w = 200
        self.bev_l = 200
        self.bev_h = 4
        self.bev_coordinate = False

        self.num_tracklets = self.init_track_id
        self.tracklets = dict()
        self.backdrops = []

    @property
    def empty(self):
        return False if self.tracklets else True

    def reset(self):
        self.trackers = list()  # tracker for each single tracklet
        self.frame_count = 0  # record for the frames
        self.time_stamp = None  # the previous time stamp

        self.num_tracklets = self.init_track_id
        self.tracklets = dict()
        self.backdrops = []
        if self.tracker_model_name == 'LSTM3DTracker':
            del self.lstm
        self.lstm = None

    def buffer_bbox_func(self, bboxes, scale_ids):
        buffered_bboxes = bboxes.clone()

        buffer_sizes = self.buffer_sizes[scale_ids]
        buffered_bboxes[:, 4:6] = buffered_bboxes[:, 4:6] * (1 + 2 * buffer_sizes.unsqueeze(-1))

        return buffered_bboxes

    def frame_mot(self, origin_data, timestamp=None):
        """ For each frame input, generate the latest mot results
        Args:
            input_data (FrameData): input data, including detection bboxes and ego information
        Returns:
            tracks on this frame: [(bbox0, id0), (bbox1, id1), ...]
        """

        raw_data = origin_data

        # input_data = FrameData(raw_data, time_stamp=self.frame_count if timestamp is None else timestamp)
        # idxes = -torch.ones_like(raw_data['labels_3d'], dtype=torch.long, device=raw_data['labels_3d'].device)
        boxes_3d_lidar_type = raw_data['boxes_3d']
        boxes_3d = deepcopy(boxes_3d_lidar_type.tensor)
        bbox_x, bbox_y, bbox_z, bbox_w, bbox_l, bbox_h, bbox_o = boxes_3d[:, 0:1], boxes_3d[:, 1:2], \
                                                                 boxes_3d[:, 2:3], boxes_3d[:, 3:4], \
                                                                 boxes_3d[:, 4:5], boxes_3d[:, 5:6], \
                                                                 boxes_3d[:, 6:7]
        bbox_z = bbox_z + bbox_h / 2
        if self.bev_coordinate:
            bbox_x = self.bev_w / 2 * (1 + bbox_x / 51.2)
            bbox_y = self.bev_l / 2 * (1 + bbox_y / 51.2)
            bbox_z = self.bev_h / 2 * (1 + (1 + bbox_z) / 4.0)

        # print(bbox_x)
        # if self.with_vxy:
        #     bbox_vx, bbox_vy = boxes_3d[:, 7:8], boxes_3d[:, 8:9]
        #     boxes_3d = torch.cat([bbox_x, bbox_y, bbox_z, bbox_o, bbox_h, bbox_w, bbox_l, bbox_vx, bbox_vy], dim=-1)
        # else:
        boxes_3d = torch.cat([bbox_x, bbox_y, bbox_z, bbox_o, bbox_h, bbox_w, bbox_l], dim=-1)

        scores_3d = raw_data['scores_3d']
        labels_3d = raw_data['labels_3d']
        # track_scores = raw_data['track_scores']
        query_feats = raw_data['query_feats']
        img_feats = raw_data['img_feats']
        bev_feats = raw_data['bev_feats']

        if 'scale_probs' in raw_data.keys():
            scale_probs = raw_data['scale_probs']
        else:
            scale_probs = torch.ones([bev_feats.size(0),5], dtype=bev_feats.dtype, device=bev_feats.device)

        self.device = query_feats.device

        # scale_ids = scale_probs.max(dim=-1)[1]
        scale_ids = torch.round((scale_probs.softmax(dim=-1) * torch.tensor(list(range(1, len(self.buffer_sizes)+1)), device=self.device)).sum(dim=-1)).to(torch.long) - 1

        if not isinstance(self.buffer_sizes, torch.Tensor):
            self.buffer_sizes = torch.tensor(self.buffer_sizes, device=self.device)
        if self.lstm is None and self.tracker_model_name == 'LSTM3DTracker':
            # self.device = torch.device(
            #     "cuda" if torch.cuda.is_available() else "cpu")
            self.lstm = get_lstm_model(self.lstm_name)(1, 64, 128, 2,
                                                       self.loc_dim).to(self.device)
            self.lstm.eval()
            ckpt = torch.load(self.lstm_ckpt_name)
            # print(f"\nLoad {self.lstm_name} checkpoints {self.lstm_ckpt_name} "
            #       f"with {ckpt['epoch']} epochs")
            try:
                self.lstm.load_state_dict(ckpt['state_dict'])
            except (RuntimeError, KeyError) as ke:
                print("Cannot load full model: {}".format(ke))
                state = self.lstm.state_dict()
                # ckpt['state_dict']["pred2vel.weight"] = ckpt['state_dict']["pred2loc.weight"]
                # del ckpt['state_dict']["pred2loc.weight"]
                state.update(ckpt['state_dict'])
                self.lstm.load_state_dict(state)
            # print(f"=> Successfully loaded checkpoint {self.lstm_ckpt_name}")
            del ckpt

        if not self.with_depth_uncertainty:
            depth_uncertainty = boxes_3d.new_ones((boxes_3d.shape[0], 1))
        else:
            depth_uncertainty = deepcopy(scores_3d)

        _, inds = depth_uncertainty.flatten().sort(descending=True)
        boxes_3d = boxes_3d[inds]
        boxes_3d_lidar_type = boxes_3d_lidar_type[inds]
        scores_3d = scores_3d[inds]
        labels_3d = labels_3d[inds]
        # track_scores = track_scores[inds]
        query_feats = query_feats[inds]
        img_feats = img_feats[inds]
        bev_feats = bev_feats[inds]
        scale_ids = scale_ids[inds]
        depth_uncertainty = depth_uncertainty[inds]

        # init ids container
        ids = torch.full((boxes_3d.size(0),), -1, dtype=torch.long, device=boxes_3d.device)

        # initialize the time stamp on frame 0
        if self.time_stamp is None:
            self.time_stamp = timestamp
            # self.time_stamp = self.frame_count

        # match if buffer is not empty
        if boxes_3d.size(0) > 0 and not self.empty:
            memo_labels_3d, memo_boxes_3d, \
            memo_trackers, memo_query_feats, memo_bev_feats, memo_img_feats, memo_ids, memo_vs, memo_scale_ids = self.memo

            memo_boxes_3d_predict = memo_boxes_3d.detach().clone()
            for ind, memo_tracker in enumerate(memo_trackers):
                memo_velo = memo_tracker.predict(
                    update_state=memo_tracker.age != 0)
                memo_boxes_3d_predict[ind, :3] += memo_boxes_3d.new_tensor(
                    memo_velo[7:])

            if self.with_bbox_iou:

                def get_xy_box(boxes_3d_world):
                    box_x_cen = boxes_3d_world[:, 0]
                    box_y_cen = boxes_3d_world[:, 1]
                    box_width = boxes_3d_world[:, 5]
                    box_length = boxes_3d_world[:, 6]

                    dets_xy_box = torch.stack([
                        box_x_cen - box_width / 2.0, box_y_cen -
                        box_length / 2.0, box_x_cen + box_width / 2.0,
                        box_y_cen + box_length / 2.0
                    ],
                        dim=1)
                    return dets_xy_box

                if self.track_bbox_iou == 'bev':
                    if self.buffer_bbox:
                        dets_xy_box = get_xy_box(self.buffer_bbox_func(boxes_3d, scale_ids))
                        memo_dets_xy_box = get_xy_box(self.buffer_bbox_func(memo_boxes_3d_predict, memo_scale_ids))
                    else:
                        dets_xy_box = get_xy_box(boxes_3d)
                        memo_dets_xy_box = get_xy_box(memo_boxes_3d_predict)
                    scores_iou = bbox_overlaps(dets_xy_box, memo_dets_xy_box)
                    # print(scores_iou)
                    scores_iou = F.sigmoid(scores_iou)
                    # print(scores_iou)
                elif self.track_bbox_iou == 'box3d':
                    depth_weight = F.pairwise_distance(
                        boxes_3d[:, None, :],
                        memo_boxes_3d_predict[None, ...])
                    scores_iou = torch.exp(-depth_weight / 10.0)
                elif self.track_bbox_iou == 'alliou':
                    dets_xy_box = get_xy_box(boxes_3d)
                    memo_dets_xy_box = get_xy_box(memo_boxes_3d_predict)
                    len_dets_xy_box = dets_xy_box.size(0)
                    len_memo_dets_xy_box = memo_dets_xy_box.size(0)
                    dets_xy_box_expand = dets_xy_box.unsqueeze(1).repeat(1, len_memo_dets_xy_box, 1).reshape(-1, 4)
                    memo_dets_xy_box_expand = memo_dets_xy_box.unsqueeze(0).repeat(len_dets_xy_box, 1, 1).reshape(-1, 4)
                    scores_iou = all_iou(dets_xy_box_expand, memo_dets_xy_box_expand, xyxy=True, DIoU=True).reshape(len_dets_xy_box, len_memo_dets_xy_box)
                    scores_iou = F.sigmoid(scores_iou)
                elif self.track_bbox_iou == 'iou3d':
                    dets_xy_box = boxes_3d[:, [0, 1, 2, 5, 4, 6, 3]]
                    memo_dets_xy_box = memo_boxes_3d_predict[:, [0, 1, 2, 5, 4, 6, 3]]
                    len_dets_xy_box = dets_xy_box.size(0)
                    len_memo_dets_xy_box = memo_dets_xy_box.size(0)
                    dets_xy_box_expand = dets_xy_box.unsqueeze(1).repeat(1, len_memo_dets_xy_box, 1).flatten(0,1).unsqueeze(0)
                    memo_dets_xy_box_expand = memo_dets_xy_box.unsqueeze(0).repeat(len_dets_xy_box, 1, 1).flatten(0,1).unsqueeze(0)
                    scores_iou = diff_iou_rotated_3d(dets_xy_box_expand, memo_dets_xy_box_expand).reshape(len_dets_xy_box, len_memo_dets_xy_box)
                    scores_iou = F.sigmoid(scores_iou)
                else:
                    raise NotImplementedError
            else:
                scores_iou = boxes_3d.new_ones(
                    [boxes_3d.size(0), memo_boxes_3d.size(0)])

            def compute_quasi_dense_feat_match(embeds, memo_embeds):
                if self.match_metric == 'cycle_softmax':
                    feats = torch.mm(embeds, memo_embeds.t())
                    d2t_scores = feats.softmax(dim=1)
                    t2d_scores = feats.softmax(dim=0)
                    scores_feat = (d2t_scores + t2d_scores) / 2
                elif self.match_metric == 'softmax':
                    feats = torch.mm(embeds, memo_embeds.t())
                    scores_feat = feats.softmax(dim=1)
                elif self.match_metric == 'norm_product':
                    scores_feat = torch.mm(
                        F.normalize(embeds, p=2, dim=1),
                        F.normalize(memo_embeds, p=2, dim=1).t())
                elif self.match_metric == 'cosine':
                    scores_feat = F.cosine_similarity(
                        embeds[:, :, None],
                        memo_embeds[:, :, None].transpose(2, 0))
                else:
                    raise NotImplementedError
                return scores_feat

            if self.with_query_feat:
                scores_query_feat = compute_quasi_dense_feat_match(
                    query_feats, memo_query_feats)
                scores_query_feat = (scores_query_feat + 1)/2
            else:
                scores_query_feat = scores_iou.new_ones(scores_iou.shape)
            if self.with_bev_feat:
                scores_bev_feat = compute_quasi_dense_feat_match(
                    bev_feats, memo_bev_feats)
                scores_bev_feat = (scores_bev_feat + 1) / 2
            else:
                scores_bev_feat = scores_iou.new_ones(scores_iou.shape)
            if self.with_img_feat:
                scores_img_feat = compute_quasi_dense_feat_match(
                    img_feats, memo_img_feats)
                scores_img_feat = (scores_img_feat + 1) / 2
            else:
                scores_img_feat = scores_iou.new_ones(scores_iou.shape)

            # Match with depth ordering
            if self.with_depth_ordering:

                def compute_boxoverlap_with_depth(obsv_boxes_3d, memo_boxes_3d,
                                                  memo_vs):
                    # Sum up all the available region of each tracker
                    if self.depth_match_metric == 'centroid':
                        depth_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., None, :3],
                            memo_boxes_3d[None, ..., :3])
                        depth_weight = torch.exp(-depth_weight / 10.0)
                    # elif self.depth_match_metric == 'cosine':
                    #     match_corners_observe = tu.worldtocamera_torch(
                    #         obsv_boxes_3d[:, :3], position, rotation)
                    #     match_corners_predict = tu.worldtocamera_torch(
                    #         memo_boxes_3d[:, :3], position, rotation)
                    #     depth_weight = F.cosine_similarity(
                    #         match_corners_observe[..., None],
                    #         match_corners_predict[..., None].transpose(2, 0))
                    #     depth_weight += 1.0
                    #     depth_weight /= 2.0
                    elif self.depth_match_metric == 'pure_motion':
                        # Moving distance should be aligned
                        # V_observed-tracked vs. V_velocity
                        depth_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., None, :3] -
                            memo_boxes_3d[None, ..., :3],
                            memo_vs[None, ..., :3])
                        depth_weight = torch.exp(-depth_weight / 5.0)
                        # Moving direction should be aligned
                        # Set to 0.5 when two vector not within +-90 degree
                        cos_sim = F.cosine_similarity(
                            obsv_boxes_3d[..., :2, None] -
                            memo_boxes_3d[..., :2, None].transpose(2, 0),
                            memo_vs[..., :2, None].transpose(2, 0))
                        cos_sim += 1.0
                        cos_sim /= 2.0
                        depth_weight *= cos_sim
                    elif self.depth_match_metric == 'motion':
                        centroid_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., None, :3],
                            memo_boxes_3d_predict[None, ..., :3])
                        centroid_weight = torch.exp(-centroid_weight / 10.0)
                        # Moving distance should be aligned
                        # V_observed-tracked vs. V_velocity
                        motion_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., None, :3] -
                            memo_boxes_3d[None, ..., :3],
                            memo_vs[None, ..., :3])
                        motion_weight = torch.exp(-motion_weight / 5.0)
                        # Moving direction should be aligned
                        # Set to 0.5 when two vector not within +-90 degree
                        cos_sim = F.cosine_similarity(
                            obsv_boxes_3d[..., :2, None] -
                            memo_boxes_3d[..., :2, None].transpose(2, 0),
                            memo_vs[..., :2, None].transpose(2, 0))
                        cos_sim += 1.0
                        cos_sim /= 2.0
                        depth_weight = cos_sim * centroid_weight + (
                                1.0 - cos_sim) * motion_weight
                    else:
                        raise NotImplementedError

                    return depth_weight

                if self.depth_match_metric == 'motion':
                    scores_depth = compute_boxoverlap_with_depth(
                        boxes_3d, memo_boxes_3d, memo_vs)
                else:
                    scores_depth = compute_boxoverlap_with_depth(
                        boxes_3d, memo_boxes_3d_predict, memo_vs)
            else:
                scores_depth = scores_iou.new_ones(scores_iou.shape)

            if self.with_cats:
                cat_same = labels_3d.view(-1, 1) == memo_labels_3d.view(1, -1)
                scores_cats = cat_same.float()
            else:
                scores_cats = scores_iou.new_ones(scores_iou.shape)

            if self.with_scales or self.multi_scale_matching:
                scale_ids_reshape = scale_ids.view(-1, 1)
                memo_scale_ids_reshape = memo_scale_ids.view(1, -1)
                scale_same = (scale_ids_reshape == (memo_scale_ids_reshape-1)) | (scale_ids_reshape == memo_scale_ids_reshape) | (scale_ids_reshape == (memo_scale_ids_reshape+1))
                scores_scales = scale_same.float()
            else:
                scores_scales = scores_iou.new_ones(scores_iou.shape)

            # scores = self.bbox_affinity_weight * scores_iou * scores_depth + \
            #          self.feat_affinity_weight * (
            #                      scores_query_feat * self.weight_query_feat + scores_bev_feat * self.weight_bev_feat + scores_img_feat * self.weight_img_feat)
            # scores /= (self.bbox_affinity_weight + self.feat_affinity_weight)
            # scores_list = [scores_iou * scores_depth, scores_query_feat * self.weight_query_feat + scores_bev_feat * self.weight_bev_feat + scores_img_feat * self.weight_img_feat]
            scores_list = [scores_query_feat * self.weight_query_feat + scores_bev_feat * self.weight_bev_feat + scores_img_feat * self.weight_img_feat, scores_iou * scores_depth]
            # scores_list = [scores_query_feat, scores_bev_feat, scores_img_feat, scores_iou * scores_depth]

            # scores_list = [scores_list[0]*self.bbox_affinity_weight + scores_list[1]*self.feat_affinity_weight]

            filters = (scores_iou > 0.0).float() * (scores_depth > 0.0).float() * scores_cats

            assert len(self.with_scales) == len(scores_list)
            assert len(self.multi_scale_matching) == len(scores_list)
            assert len(self.match_score_thr) == len(scores_list) if isinstance(self.match_score_thr, list) else True

            scores_list = [scores * scores_scales if with_scale else scores for with_scale, scores in zip(self.with_scales, scores_list)]

            scores_list = torch.stack([scores * filters for scores in scores_list], dim=0)

            num_det = boxes_3d.size(0)
            recoder = torch.ones(num_det, device=self.device)

            # Assign matching
            for scores_idx in range(scores_list.size(0)):
                scores = scores_list[scores_idx]

                if self.multi_scale_matching[scores_idx]:
                    for scale_idx in reversed(range(len(self.buffer_sizes))):
                        scale_select = (scale_ids == scale_idx).float()
                        remain_idx = torch.nonzero(recoder * scale_select).flatten()
                        if len(remain_idx) == 0:
                            continue
                        map_dict = {k: v for k, v in enumerate(remain_idx)}

                        scores_remain = scores[remain_idx]
                        if self.match_algo == 'greedy':
                            for i in range(len(remain_idx)):
                                conf, memo_ind = torch.max(scores_remain[i, :], dim=0)
                                tid = memo_ids[memo_ind]
                                # Matching confidence
                                if conf > self.match_score_thr[scores_idx] if isinstance(self.match_score_thr, list) else self.match_score_thr[scores_idx]:
                                    # Update existing tracklet
                                    if tid > -1:
                                        recoder[map_dict[i]] = 0
                                        # Keep object with high 3D objectness
                                        if depth_uncertainty[map_dict[i]] > self.obj_score_thr:
                                            ids[map_dict[i]] = tid
                                            scores_list[:, map_dict[i], memo_ind] = 0
                                            scores_list[:, map_dict[i] + 1:, memo_ind] = 0
                                        else:
                                            # Reduce FP w/ low objectness but high match conf
                                            if conf > self.nms_conf_thr:
                                                ids[map_dict[i]] = -2
                        elif self.match_algo == 'hungarian':
                            # Hungarian
                            matched_indices = linear_assignment(-scores_remain.cpu().numpy())
                            for idx in range(len(matched_indices[0])):
                                i = matched_indices[0][idx]
                                memo_ind = matched_indices[1][idx]
                                conf = scores_remain[i, memo_ind]
                                tid = memo_ids[memo_ind]
                                if (conf > self.match_score_thr[scores_idx] if isinstance(self.match_score_thr, list) else self.match_score_thr[scores_idx]) and tid > -1:
                                    recoder[map_dict[i]] = 0
                                    # Keep object with high 3D objectness
                                    if depth_uncertainty[map_dict[i]] > self.obj_score_thr:
                                        ids[map_dict[i]] = tid
                                        scores_list[:, map_dict[i], memo_ind] = 0
                                        scores_list[:, map_dict[i] + 1:, memo_ind] = 0
                                    else:
                                        # Reduce FP w/ low objectness but high match conf
                                        if conf > self.nms_conf_thr:
                                            ids[map_dict[i]] = -2
                            del matched_indices
                    if self.re_matching:
                        remain_idx = torch.nonzero(recoder).flatten()
                        map_dict = {k: v for k, v in enumerate(remain_idx)}

                        scores_remain = scores[remain_idx]
                        if self.match_algo == 'greedy':
                            for i in range(len(remain_idx)):
                                conf, memo_ind = torch.max(scores_remain[i, :], dim=0)
                                tid = memo_ids[memo_ind]
                                # Matching confidence
                                if conf > self.match_score_thr[scores_idx] if isinstance(self.match_score_thr, list) else self.match_score_thr[scores_idx]:
                                    # Update existing tracklet
                                    if tid > -1:
                                        recoder[map_dict[i]] = 0
                                        # Keep object with high 3D objectness
                                        if depth_uncertainty[map_dict[i]] > self.obj_score_thr:
                                            ids[map_dict[i]] = tid
                                            scores_list[:, map_dict[i], memo_ind] = 0
                                            scores_list[:, map_dict[i] + 1:, memo_ind] = 0
                                        else:
                                            # Reduce FP w/ low objectness but high match conf
                                            if conf > self.nms_conf_thr:
                                                ids[map_dict[i]] = -2
                        elif self.match_algo == 'hungarian':
                            # Hungarian
                            matched_indices = linear_assignment(-scores_remain.cpu().numpy())
                            for idx in range(len(matched_indices[0])):
                                i = matched_indices[0][idx]
                                memo_ind = matched_indices[1][idx]
                                conf = scores_remain[i, memo_ind]
                                tid = memo_ids[memo_ind]
                                if (conf > self.match_score_thr[scores_idx] if isinstance(self.match_score_thr, list) else self.match_score_thr[scores_idx]) and tid > -1:
                                    recoder[map_dict[i]] = 0
                                    # Keep object with high 3D objectness
                                    if depth_uncertainty[map_dict[i]] > self.obj_score_thr:
                                        ids[map_dict[i]] = tid
                                        scores_list[:, map_dict[i], memo_ind] = 0
                                        scores_list[:, map_dict[i] + 1:, memo_ind] = 0
                                    else:
                                        # Reduce FP w/ low objectness but high match conf
                                        if conf > self.nms_conf_thr:
                                            ids[map_dict[i]] = -2
                            del matched_indices
                else:
                    remain_idx = torch.nonzero(recoder).flatten()
                    map_dict = {k: v for k, v in enumerate(remain_idx)}

                    scores_remain = scores[remain_idx]
                    if self.match_algo == 'greedy':
                        for i in range(len(remain_idx)):
                            conf, memo_ind = torch.max(scores_remain[i, :], dim=0)
                            tid = memo_ids[memo_ind]
                            # Matching confidence
                            if conf > self.match_score_thr[scores_idx] if isinstance(self.match_score_thr, list) else self.match_score_thr[scores_idx]:
                                # Update existing tracklet
                                if tid > -1:
                                    recoder[map_dict[i]] = 0
                                    # Keep object with high 3D objectness
                                    if depth_uncertainty[map_dict[i]] > self.obj_score_thr:
                                        ids[map_dict[i]] = tid
                                        scores_list[:, map_dict[i], memo_ind] = 0
                                        scores_list[:, map_dict[i] + 1:, memo_ind] = 0
                                    else:
                                        # Reduce FP w/ low objectness but high match conf
                                        if conf > self.nms_conf_thr:
                                            ids[map_dict[i]] = -2
                    elif self.match_algo == 'hungarian':
                        # Hungarian
                        matched_indices = linear_assignment(-scores_remain.cpu().numpy())
                        for idx in range(len(matched_indices[0])):
                            i = matched_indices[0][idx]
                            memo_ind = matched_indices[1][idx]
                            conf = scores_remain[i, memo_ind]
                            tid = memo_ids[memo_ind]
                            if (conf > self.match_score_thr[scores_idx] if isinstance(self.match_score_thr, list) else self.match_score_thr[scores_idx]) and tid > -1:
                                recoder[map_dict[i]] = 0
                                # Keep object with high 3D objectness
                                if depth_uncertainty[map_dict[i]] > self.obj_score_thr:
                                    ids[map_dict[i]] = tid
                                    scores_list[:, map_dict[i], memo_ind] = 0
                                    scores_list[:, map_dict[i] + 1:, memo_ind] = 0
                                else:
                                    # Reduce FP w/ low objectness but high match conf
                                    if conf > self.nms_conf_thr:
                                        ids[map_dict[i]] = -2
                        del matched_indices

        new_inds = (ids == -1) & (scores_3d > self.init_score_thr)
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracklets,
            self.num_tracklets + num_news,
            dtype=torch.long, device=self.device)
        self.num_tracklets += num_news

        # self.frame_count += 1
        self.update_memo(ids, boxes_3d, depth_uncertainty, query_feats, bev_feats, img_feats,
                         labels_3d, scale_ids, cur_frame=self.frame_count)
        self.frame_count += 1

        update_scores_3d = scores_3d.detach().clone()
        update_labels_3d = labels_3d.detach().clone()

        # choice: replace
        # update_boxes_3d = boxes_3d.detach().clone()
        # for tid in ids[ids > -1]:
        #     update_boxes_3d[ids == tid] = self.tracklets[int(tid)]['box_3d']
        # update_boxes_3d[2] -= update_boxes_3d[4] / 2
        # boxes_3d_lidar_type_tensor = boxes_3d_lidar_type.tensor
        # boxes_3d_lidar_type.tensor = torch.cat([update_boxes_3d[:, [0,1,2,5,6,4,3]], boxes_3d_lidar_type_tensor[:, -2:]], dim=-1)

        update_boxes_3d = boxes_3d_lidar_type

        update_ids = ids.detach().clone()

        ids_mask = update_ids > -1

        # result_dict = dict(
        #     boxes_3d=update_boxes_3d.to('cpu'),
        #     scores_3d=update_scores_3d.cpu(),
        #     labels_3d=update_labels_3d.cpu(),
        #     track_scores=update_scores_3d.cpu(),
        #     track_ids=update_ids.cpu(),
        # )
        result_dict = dict(
            boxes_3d=update_boxes_3d[ids_mask].to('cpu'),
            scores_3d=update_scores_3d[ids_mask].cpu(),
            labels_3d=update_labels_3d[ids_mask].cpu(),
            track_scores=update_scores_3d[ids_mask].cpu(),
            track_ids=update_ids[ids_mask].cpu(),
        )
        # print(update_ids)

        return result_dict

    def update_memo(self, ids, boxes_3d, depth_uncertainty, query_feats, bev_feats, img_feats, labels_3d, scale_ids, cur_frame):
        tracklet_inds = ids > -1

        # update memo
        for tid, box_3d, d_uncertainty, query_feat, bev_feat, img_feat, label_3d, scale_id in zip(
                ids[tracklet_inds], boxes_3d[tracklet_inds], depth_uncertainty[tracklet_inds],
                query_feats[tracklet_inds], bev_feats[tracklet_inds],
                img_feats[tracklet_inds], labels_3d[tracklet_inds], scale_ids[tracklet_inds]):
            tid = int(tid)
            if tid in self.tracklets.keys():
                # self.tracklets[tid]['bbox'] = bbox

                # self.tracklets[tid]['tracker'].update(box_3d.cpu().numpy(), d_uncertainty.cpu().numpy())
                self.tracklets[tid]['tracker'].update(box_3d, d_uncertainty)

                tracker_box = self.tracklets[tid]['tracker'].get_state()[:7]
                pd_box_3d = box_3d.new_tensor(tracker_box)

                velocity = (pd_box_3d - self.tracklets[tid]['box_3d']) / (
                    cur_frame - self.tracklets[tid]['last_frame'])

                self.tracklets[tid]['box_3d'] = pd_box_3d
                self.tracklets[tid]['query_feat'] += self.memo_momentum * (query_feat - self.tracklets[tid]['query_feat'])
                self.tracklets[tid]['bev_feat'] += self.memo_momentum * (bev_feat - self.tracklets[tid]['bev_feat'])
                self.tracklets[tid]['img_feat'] += self.memo_momentum * (img_feat - self.tracklets[tid]['img_feat'])
                self.tracklets[tid]['label_3d'] = label_3d
                self.tracklets[tid]['scale_id'] = scale_id
                self.tracklets[tid]['velocity'] = (
                    self.tracklets[tid]['velocity'] *
                    self.tracklets[tid]['acc_frame'] + velocity) / (
                        self.tracklets[tid]['acc_frame'] + 1)
                self.tracklets[tid]['last_frame'] = cur_frame
                self.tracklets[tid]['acc_frame'] += 1
            else:
                built_tracker = self.tracker_model(
                    self.lstm,
                    box_3d,  # box_3d.cpu().numpy(),
                    d_uncertainty,  # d_uncertainty.cpu().numpy()
                ) if self.tracker_model_name == 'LSTM3DTracker' else self.tracker_model(
                    box_3d,  # box_3d.cpu().numpy(),
                    d_uncertainty,  # d_uncertainty.cpu().numpy()
                )
                self.tracklets[tid] = dict(
                    # bbox=bbox,
                    box_3d=box_3d,
                    tracker=built_tracker,
                    query_feat=query_feat,
                    bev_feat=bev_feat,
                    img_feat=img_feat,
                    label_3d=label_3d,
                    scale_id=scale_id,
                    last_frame=cur_frame,
                    velocity=torch.zeros_like(box_3d, device=boxes_3d.device),
                    acc_frame=0)

        # Handle vanished tracklets
        for tid in self.tracklets:
            if cur_frame > self.tracklets[tid]['last_frame'] and tid > -1:
                self.tracklets[tid]['box_3d'][:self.loc_dim] = self.tracklets[
                    tid]['box_3d'].new_tensor(
                        self.tracklets[tid]['tracker'].predict()
                        [:self.loc_dim])

        # Add backdrops
        backdrop_inds = torch.nonzero(ids == -1).squeeze(1)
        def get_xy_box(boxes_3d_world):
            box_x_cen = boxes_3d_world[:, 0]
            box_y_cen = boxes_3d_world[:, 1]
            box_width = boxes_3d_world[:, 5]
            box_length = boxes_3d_world[:, 6]

            dets_xy_box = torch.stack([
                box_x_cen - box_width / 2.0, box_y_cen -
                box_length / 2.0, box_x_cen + box_width / 2.0,
                box_y_cen + box_length / 2.0
            ],
                dim=1)
            return dets_xy_box

        ious = bbox_overlaps(get_xy_box(boxes_3d[backdrop_inds]), get_xy_box(boxes_3d))
        # ious = bbox_overlaps(boxes_3d[backdrop_inds, :-1], boxes_3d[:, :-1])
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        backdrop_trackers = [
            self.tracker_model(self.lstm,
                               boxes_3d[bd_ind],  # boxes_3d[bd_ind].cpu().numpy(),
                               depth_uncertainty[bd_ind],  # depth_uncertainty[bd_ind].cpu().numpy()
                               ) if
            self.tracker_model_name == 'LSTM3DTracker' else self.tracker_model(
                boxes_3d[bd_ind],  # boxes_3d[bd_ind].cpu().numpy(),
                depth_uncertainty[bd_ind],  # depth_uncertainty[bd_ind].cpu().numpy()
            )
            for bd_ind in backdrop_inds
        ]

        self.backdrops.insert(
            0,
            dict(
                # bboxes=bboxes[backdrop_inds],
                boxes_3d=boxes_3d[backdrop_inds],
                trackers=backdrop_trackers,
                query_feats=query_feats[backdrop_inds],
                bev_feats=bev_feats[backdrop_inds],
                img_feats=img_feats[backdrop_inds],
                labels_3d=labels_3d[backdrop_inds],
                scale_id=scale_ids[backdrop_inds],
            ))

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if cur_frame - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self):
        memo_query_feats = []
        memo_bev_feats = []
        memo_img_feats = []
        memo_ids = []
        # memo_bboxes = []
        memo_boxes_3d = []
        memo_trackers = []
        memo_labels_3d = []
        memo_scale_ids = []
        memo_vs = []
        for k, v in self.tracklets.items():
            # memo_bboxes.append(v['bbox'][None, :])
            memo_boxes_3d.append(v['box_3d'][None, :])
            memo_trackers.append(v['tracker'])
            memo_query_feats.append(v['query_feat'][None, :])
            memo_bev_feats.append(v['bev_feat'][None, :])
            memo_img_feats.append(v['img_feat'][None, :])
            memo_ids.append(k)
            memo_labels_3d.append(v['label_3d'].view(1, 1))
            memo_scale_ids.append(v['scale_id'].view(1, 1))
            memo_vs.append(v['velocity'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long, device=memo_query_feats[-1].device).view(1, -1)

        for backdrop in self.backdrops:
            backdrop_ids = torch.full((1, backdrop['boxes_3d'].size(0)),
                                      -1,
                                      dtype=torch.long, device=backdrop['boxes_3d'].device)
            backdrop_vs = torch.zeros_like(backdrop['boxes_3d'], device=backdrop['boxes_3d'].device)
            # memo_bboxes.append(backdrop['bboxes'])
            memo_boxes_3d.append(backdrop['boxes_3d'])
            memo_trackers.extend(backdrop['trackers'])
            memo_query_feats.append(backdrop['query_feats'])
            memo_bev_feats.append(backdrop['bev_feats'])
            memo_img_feats.append(backdrop['img_feats'])
            memo_ids = torch.cat([memo_ids, backdrop_ids], dim=1)
            memo_labels_3d.append(backdrop['labels_3d'][:, None])
            memo_scale_ids.append(backdrop['scale_id'][:, None])
            memo_vs.append(backdrop_vs)

        # memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_boxes_3d = torch.cat(memo_boxes_3d, dim=0)
        memo_query_feats = torch.cat(memo_query_feats, dim=0)
        memo_bev_feats = torch.cat(memo_bev_feats, dim=0)
        memo_img_feats = torch.cat(memo_img_feats, dim=0)
        memo_labels_3d = torch.cat(memo_labels_3d, dim=0).squeeze(1)
        memo_scale_ids = torch.cat(memo_scale_ids, dim=0).squeeze(1)
        memo_vs = torch.cat(memo_vs, dim=0)
        return memo_labels_3d, memo_boxes_3d, memo_trackers, memo_query_feats, memo_bev_feats, memo_img_feats, memo_ids.squeeze(
            0), memo_vs, memo_scale_ids
