import torch
import torch.nn.functional as F
import numpy as np, models.trackers_qd3dt.tracklet as tracklet
from mmdet3d.models.losses import axis_aligned_iou_loss
from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_3d, bbox_overlaps_nearest_3d
from . import utils
from scipy.optimize import linear_sum_assignment
from .frame_data import FrameData
from .update_info_data import UpdateInfoData
from .bbox import BBox, Validity

from mmcv.ops import diff_iou_rotated_3d
# from cython_bbox import bbox_overlaps as bbox_ious


def compute_embed_distance(dets, tracks, mode='cos'):
# def compute_embed_distance(dets, tracks, mode='cdist'):
    len_dets = len(dets)
    len_tracks = len(tracks)
    # print(len_dets, len_tracks)
    if len_dets == 0 or len_tracks == 0:
        return np.ones((len_dets, len_tracks)), np.zeros((len_dets, len_tracks)).astype(np.bool_)
    det_embeds = []
    det_labels = []
    for d, det in enumerate(dets):
        det_embeds.append(det.query_feat)
        det_labels.append(det.label)

    det_labels = torch.stack(det_labels)

    track_embeds = []
    track_labels = []
    for d, dtrket in enumerate(tracks):
        track_embeds.append(dtrket.query_feat)
        dtrket_class_count = dtrket.class_count
        value, indexes = torch.topk(dtrket_class_count, k=2)
        track_labels.append((det_labels == indexes[0]) | (det_labels == indexes[1]))

    dim = dtrket.query_feat.size(-1)
    det_embeds = torch.stack(det_embeds, dim=0)
    track_embeds = torch.stack(track_embeds, dim=0)

    track_labels = torch.stack(track_labels, dim=1)
    # print(det_labels.shape, track_labels.shape)
    # print(det_embeds.shape, track_embeds.shape)

    if mode == 'cos':
        det_embeds = det_embeds.unsqueeze(1).repeat(1, len_tracks, 1).reshape(-1, dim)
        track_embeds = track_embeds.unsqueeze(0).repeat(len_dets, 1, 1).reshape(-1, dim)
        embeds_matrix = F.cosine_similarity(det_embeds, track_embeds).reshape(len_dets, len_tracks)
    elif mode == 'cdist':
        embeds_matrix = torch.cdist(det_embeds, track_embeds, p=2)

    embeds_matrix = F.sigmoid(embeds_matrix)
    # embeds_matrix = (1+embeds_matrix) / 2
    # embeds_matrix = torch.clamp(embeds_matrix, min=0.)
    # embeds_matrix = torch.abs(embeds_matrix)

    # print(embeds_matrix)
    return 1 - embeds_matrix.cpu(), track_labels

def embed_match(dets, tracks, mode='bipartite', dist_threshold=None):
    dist_matrix, dist_matrix_mask = compute_embed_distance(dets, tracks)

    dist_matrix[~dist_matrix_mask] = 10
    # dist_matrix[dist_matrix>dist_threshold] = 10

    if mode == 'bipartite':
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        # print(dist_matrix.shape, '2')
        matched_indices = np.stack([row_ind, col_ind], axis=1)
    elif mode == 'greedy':
        matched_indices = list()
        num_dets, num_trks = dist_matrix.shape

        # association in the greedy manner
        # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py
        distance_1d = dist_matrix.reshape(-1)
        index_1d = np.argsort(distance_1d)
        index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)
        detection_id_matches_to_tracking_id = [-1] * num_dets
        tracking_id_matches_to_detection_id = [-1] * num_trks
        for sort_i in range(index_2d.shape[0]):
            detection_id = int(index_2d[sort_i][0])
            tracking_id = int(index_2d[sort_i][1])
            if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[
                detection_id] == -1:
                tracking_id_matches_to_detection_id[tracking_id] = detection_id
                detection_id_matches_to_tracking_id[detection_id] = tracking_id
                matched_indices.append([detection_id, tracking_id])
        if len(matched_indices) == 0:
            matched_indices = np.empty((0, 2))
        else:
            matched_indices = np.asarray(matched_indices)
    return matched_indices, dist_matrix, dist_matrix_mask

# def associate_dets_to_tracks_forward(dets, tracks, mode, asso, dist_threshold=0.9, trk_innovation_matrix=None, embed_threshold=0.7):
# def associate_dets_to_tracks(dets, tracks, mode, asso, dist_threshold=0.9, trk_innovation_matrix=None, embed_threshold=0.7):
def associate_dets_to_tracks(dets, tracks, mode, asso, dist_threshold=0.9, trk_innovation_matrix=None, embed_threshold=0.3): # 0.4-0.38-0.334 0.3-0.7-0.38-0.342 0.3-0.6-0.38-0.343
    """ associate the tracks with detections
    """
    matches = list()
    matched_indices, dist_matrix, dist_matrix_mask = embed_match(dets, tracks, dist_threshold=embed_threshold)
    unmatched_dets = list()
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)
    unmatched_tracks = list()
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > embed_threshold:
        # if dist_matrix[m[0], m[1]] > embed_threshold or dist_matrix_mask[m[0], m[1]] is False:
            unmatched_dets.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(2))
    # print(unmatched_dets)
    # dets = dets[unmatched_dets]
    # tracks = tracks[unmatched_tracks]
    dets = [dets[idx] for idx in unmatched_dets]
    tracks = [tracks[idx] for idx in unmatched_tracks]
    map_dets = {k: v for k, v in enumerate(unmatched_dets)}
    map_tracks = {k: v for k, v in enumerate(unmatched_tracks)}


    if mode == 'bipartite':
        matched_indices, dist_matrix = \
            bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    elif mode == 'greedy':
        matched_indices, dist_matrix = \
            greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    unmatched_dets = list()
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(map_dets[d])

    unmatched_tracks = list()
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(map_tracks[t])

    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > dist_threshold:
            unmatched_dets.append(map_dets[m[0]])
            unmatched_tracks.append(map_tracks[m[1]])
        else:
            matches.append([map_dets[m[0]], map_tracks[m[1]]])
    return matches, np.array(unmatched_dets), np.array(unmatched_tracks)


def associate_dets_to_tracks_backward(dets, tracks, mode, asso, dist_threshold=0.9, trk_innovation_matrix=None, embed_threshold=0.7):
# def associate_dets_to_tracks(dets, tracks, mode, asso, dist_threshold=0.9, trk_innovation_matrix=None, embed_threshold=0.7):
    """ associate the tracks with detections
    """
    matches = list()

    if mode == 'bipartite':
        matched_indices, dist_matrix = \
            bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    elif mode == 'greedy':
        matched_indices, dist_matrix = \
            greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    unmatched_dets = list()
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    unmatched_tracks = list()
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)

    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > dist_threshold:
            unmatched_dets.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(2))


    dets = [dets[idx] for idx in unmatched_dets]
    tracks = [tracks[idx] for idx in unmatched_tracks]
    map_dets = {k: v for k, v in enumerate(unmatched_dets)}
    map_tracks = {k: v for k, v in enumerate(unmatched_tracks)}
    matched_indices, dist_matrix = embed_match(dets, tracks)
    unmatched_dets = list()
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(map_dets[d])
    unmatched_tracks = list()
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(map_tracks[t])
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > embed_threshold:
            unmatched_dets.append(map_dets[m[0]])
            unmatched_tracks.append(map_tracks[m[1]])
        else:
            matches.append([map_dets[m[0]], map_tracks[m[1]]])
    # print(unmatched_dets)
    # dets = dets[unmatched_dets]
    # tracks = tracks[unmatched_tracks]
    return matches, np.array(unmatched_dets), np.array(unmatched_tracks)

def associate_dets_to_tracks_ago(dets, tracks, mode, asso,
                             dist_threshold=0.9, trk_innovation_matrix=None):
    """ associate the tracks with detections
    """
    matches = list()

    if mode == 'bipartite':
        matched_indices, dist_matrix = \
            bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    elif mode == 'greedy':
        matched_indices, dist_matrix = \
            greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    unmatched_dets = list()
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    unmatched_tracks = list()
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)


    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > dist_threshold:
            unmatched_dets.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(2))
    return matches, np.array(unmatched_dets), np.array(unmatched_tracks)


def bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix):
    if asso == 'iou':
        dist_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'giou':
        dist_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'm_dis':
        dist_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        dist_matrix = compute_m_distance(dets, tracks, None)
    # print(dist_matrix.shape, '1')
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    # print(dist_matrix.shape, '2')
    matched_indices = np.stack([row_ind, col_ind], axis=1)
    return matched_indices, dist_matrix


def greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix):
    """ it's ok to use iou in bipartite
        but greedy is only for m_distance
    """
    matched_indices = list()

    # compute the distance matrix
    if asso == 'm_dis':
        distance_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        distance_matrix = compute_m_distance(dets, tracks, None)
    elif asso == 'iou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'giou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    num_dets, num_trks = distance_matrix.shape

    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_dets
    tracking_id_matches_to_detection_id = [-1] * num_trks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[
            detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])
    if len(matched_indices) == 0:
        matched_indices = np.empty((0, 2))
    else:
        matched_indices = np.asarray(matched_indices)
    return matched_indices, distance_matrix


# def compute_m_distance(dets, tracks, trk_innovation_matrix):
#     """ compute l2 or mahalanobis distance
#         when the input trk_innovation_matrix is None, compute L2 distance (euler)
#         else compute mahalanobis distance
#         return dist_matrix: numpy array [len(dets), len(tracks)]
#     """
#     euler_dis = (trk_innovation_matrix is None)  # is use euler distance
#     if not euler_dis:
#         trk_inv_inn_matrices = [np.linalg.inv(m) for m in trk_innovation_matrix]
#     dist_matrix = np.empty((len(dets), len(tracks)))
#
#     for i, det in enumerate(dets):
#         for j, trk in enumerate(tracks):
#             if euler_dis:
#                 dist_matrix[i, j] = utils.m_distance(det, trk)
#             else:
#                 dist_matrix[i, j] = utils.m_distance(det, trk, trk_inv_inn_matrices[j])
#     return dist_matrix

def compute_m_distance(dets, tracks, trk_innovation_matrix):
    """ compute l2 or mahalanobis distance
        when the input trk_innovation_matrix is None, compute L2 distance (euler)
        else compute mahalanobis distance
        return dist_matrix: numpy array [len(dets), len(tracks)]
    """
    euler_dis = (trk_innovation_matrix is None)  # is use euler distance
    if not euler_dis:
        trk_inv_inn_matrices = [torch.linalg.inv(m) for m in trk_innovation_matrix]
    dist_matrix = np.empty((len(dets), len(tracks)))

    for i, det in enumerate(dets):
        for j, trk in enumerate(tracks):
            if euler_dis:
                dist_matrix[i, j] = utils.m_distance(det, trk)
            else:
                dist_matrix[i, j] = utils.m_distance(det, trk, trk_inv_inn_matrices[j])
    return dist_matrix

# def compute_iou_distance(dets, tracks, asso='iou'):
#     iou_matrix = np.zeros((len(dets), len(tracks)))
#     for d, det in enumerate(dets):
#         for t, trk in enumerate(tracks):
#             if asso == 'iou':
#                 iou_matrix[d, t] = utils.iou3d(det, trk)[1]
#             elif asso == 'giou':
#                 iou_matrix[d, t] = utils.giou3d(det, trk)
#     dist_matrix = 1 - iou_matrix
#     return dist_matrix



# bbox [cx, cy, bottom-z, w, l, h, rotation]
# iou = bbox_overlaps_3d(bboxes1, bboxes2, mode='iou', coordinate='lidar')  # camera or lidar
# iou = bbox_overlaps_nearest_3d(bboxes1, bboxes2, mode='iou', coordinate='depth')  # camera, depth, or lidar

# def compute_iou_distance(dets, tracks, asso='iou'):
#     if len(tracks) == 0 or len(dets) == 0:
#         return np.ones((len(dets), len(tracks)))
#     det_bboxes = []
#     for d, det in enumerate(dets):
#         det_bboxes.append(BBox.bbox2array_lidar(det))
#     det_bboxes = torch.stack(det_bboxes, dim=0)
#     track_bboxes = []
#     for d, dtrket in enumerate(tracks):
#         track_bboxes.append(BBox.bbox2array_lidar(dtrket))
#     track_bboxes = torch.stack(track_bboxes, dim=0)
#     # iou_matrix = bbox_overlaps_3d(det_bboxes, track_bboxes, mode='iou', coordinate='lidar')  # camera or lidar
#     iou_matrix = bbox_overlaps_nearest_3d(det_bboxes, track_bboxes, mode='iou', coordinate='depth')  # camera or lidar
#     # print(torch.sum(iou_matrix>0.5))
#     dist_matrix = 1 - iou_matrix
#     # print(dist_matrix)
#     return dist_matrix.cpu()

def compute_iou_distance(dets, tracks, asso='iou'):
    len_dets = len(dets)
    len_tracks = len(tracks)
    if len_dets == 0 or len_tracks == 0:
        return np.ones((len_dets, len_tracks))
    det_bboxes = []
    for d, det in enumerate(dets):
        det_bboxes.append(BBox.bbox2array_lidar_center(det))
    track_bboxes = []
    for d, dtrket in enumerate(tracks):
        track_bboxes.append(BBox.bbox2array_lidar_center(dtrket))

    # center l1
    # det_bboxes = torch.abs(torch.stack(det_bboxes, dim=0))[:, [0,1]] / 51.2
    # track_bboxes = torch.abs(torch.stack(track_bboxes, dim=0))[:, [0,1]] / 51.2
    # dist_matrix = torch.cdist(det_bboxes, track_bboxes, p=1)/2
    # # det_bboxes = torch.abs(torch.stack(det_bboxes, dim=0))[:, [0, 1, 3, 4]] / 51.2
    # # track_bboxes = torch.abs(torch.stack(track_bboxes, dim=0))[:, [0, 1, 3, 4]] / 51.2
    # # dist_matrix = torch.cdist(det_bboxes, track_bboxes, p=1) / 4
    # dist_matrix = F.sigmoid(dist_matrix)

    # print(det_bboxes.shape, track_bboxes.shape)
    # iou_matrix = bbox_overlaps_3d(det_bboxes, track_bboxes, mode='iou', coordinate='lidar')  # camera or lidar
    # iou_matrix = bbox_overlaps_nearest_3d(det_bboxes, track_bboxes, mode='iou', coordinate='depth')  # camera or lidar
    det_bboxes = torch.stack(det_bboxes, dim=0).unsqueeze(1).repeat(1, len_tracks, 1).reshape(1, -1, 7)
    track_bboxes = torch.stack(track_bboxes, dim=0).unsqueeze(0).repeat(len_dets, 1, 1).reshape(1, -1, 7)
    iou_matrix = diff_iou_rotated_3d(det_bboxes, track_bboxes).reshape(len_dets, len_tracks)
    # print(torch.sum(iou_matrix>0.5))
    dist_matrix = 1 - iou_matrix
    # print(dist_matrix)

    return dist_matrix.cpu()

def compute_iou_distance_now(dets, tracks, asso='iou'):
    len_dets = len(dets)
    len_tracks = len(tracks)
    if len_dets == 0 or len_tracks == 0:
        return np.ones((len_dets, len_tracks))
    det_bboxes = []
    for d, det in enumerate(dets):
        det_bboxes.append(BBox.bbox2array_lidar_center(det))
    track_bboxes = []
    for d, dtrket in enumerate(tracks):
        track_bboxes.append(BBox.bbox2array_lidar_center(dtrket))

    # center l1
    # det_bboxes = torch.abs(torch.stack(det_bboxes, dim=0))[:, [0,1]] / 51.2
    # track_bboxes = torch.abs(torch.stack(track_bboxes, dim=0))[:, [0,1]] / 51.2
    # dist_matrix = torch.cdist(det_bboxes, track_bboxes, p=1)/2
    # # det_bboxes = torch.abs(torch.stack(det_bboxes, dim=0))[:, [0, 1, 3, 4]] / 51.2
    # # track_bboxes = torch.abs(torch.stack(track_bboxes, dim=0))[:, [0, 1, 3, 4]] / 51.2
    # # dist_matrix = torch.cdist(det_bboxes, track_bboxes, p=1) / 4
    # dist_matrix = F.sigmoid(dist_matrix)

    # print(det_bboxes.shape, track_bboxes.shape)
    # iou_matrix = bbox_overlaps_3d(det_bboxes, track_bboxes, mode='iou', coordinate='lidar')  # camera or lidar
    # iou_matrix = bbox_overlaps_nearest_3d(det_bboxes, track_bboxes, mode='iou', coordinate='depth')  # camera or lidar
    det_bboxes = torch.stack(det_bboxes, dim=0)[:, [0,1,3,4]].unsqueeze(1).repeat(1, len_tracks, 1).reshape(-1, 4)
    track_bboxes = torch.stack(track_bboxes, dim=0)[:, [0,1,3,4]].unsqueeze(0).repeat(len_dets, 1, 1).reshape(-1, 4)
    # det_bboxes[:, 0] -= det_bboxes[:, 2] / 2
    # det_bboxes[:, 1] -= det_bboxes[:, 3] / 2
    # det_bboxes[:, 2] += det_bboxes[:, 0]
    # det_bboxes[:, 3] += det_bboxes[:, 1]
    # track_bboxes[:, 0] -= track_bboxes[:, 2] / 2
    # track_bboxes[:, 1] -= track_bboxes[:, 3] / 2
    # track_bboxes[:, 2] += track_bboxes[:, 0]
    # track_bboxes[:, 3] += track_bboxes[:, 1]
    iou_matrix = all_iou(det_bboxes, track_bboxes, xywh=True, IoU=True).reshape(len_dets, len_tracks)
    # print(iou_matrix.shape, iou_matrix)
    # iou_matrix = diff_iou_rotated_3d(det_bboxes, track_bboxes).reshape(len_dets, len_tracks)
    # print(torch.sum(iou_matrix>0.5))
    dist_matrix = 1 - iou_matrix
    # print(dist_matrix)

    return dist_matrix.cpu()


def all_iou(box1, box2, weight=None, xyxy=False, xywh=False, ltrb=False, IoU=False, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    # box1 = box1.t()
    # box2 = box2.t()
    # loss_bbox = F.l1_loss(box1, box2, reduction='none').sum(dim=-1)

    # Get the coordinates of bounding boxes
    if ltrb:
        b1_x1 = 128.0 - box1[:, 0]
        b1_x2 = 128.0 + box1[:, 2]
        b1_y1 = 128.0 - box1[:, 1]
        b1_y2 = 128.0 + box1[:, 3]
        b2_x1 = 128.0 - box2[:, 0]
        b2_x2 = 128.0 + box2[:, 2]
        b2_y1 = 128.0 - box2[:, 1]
        b2_y2 = 128.0 + box2[:, 3]
    elif xywh:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    elif xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if IoU:
        return iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            losses = iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                losses = iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / np.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                losses = iou - (rho2 / c2 + v * alpha)  # CIoU
    else:
        losses = iou
    # losses = -torch.log((1.-losses)/2.)
    # losses = 1. - losses

    # weight_sum = weight.sum()
    # losses = (losses * weight).sum() / weight_sum
    return losses
