""" The defination and basic methods of bbox
"""
import torch
import numpy as np
from copy import deepcopy


class BBox:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, o=None):
        self.x = x  # center x
        self.y = y  # center y
        self.z = z  # center z
        self.h = h  # height
        self.w = w  # width
        self.l = l  # length
        self.o = o  # orientation
        self.s = None  # detection score

    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.o, self.l, self.w, self.h, self.s)

    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.o}

    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return torch.stack([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h])
        else:
            return torch.stack([bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def bbox2array_lidar(cls, bbox):
        if bbox.s is None:
            return torch.stack([bbox.x, bbox.y, bbox.z - bbox.h / 2, bbox.l, bbox.w, bbox.h, bbox.o])
        else:
            return torch.stack([bbox.x, bbox.y, bbox.z - bbox.h / 2, bbox.l, bbox.w, bbox.h, bbox.o, bbox.s])

    @classmethod
    def bbox2array_lidar_center(cls, bbox):
        return torch.stack([bbox.x, bbox.y, bbox.z, bbox.l, bbox.w, bbox.h, bbox.o])
        # if bbox.s is None:
        #     return torch.stack([bbox.x, bbox.y, bbox.z, bbox.l, bbox.w, bbox.h, bbox.o])
        # else:
        #     return torch.stack([bbox.x, bbox.y, bbox.z, bbox.l, bbox.w, bbox.h, bbox.o, bbox.s])

    @classmethod
    def array2bbox_init(cls, data, score_3d, label_3d, query_feat, img_feat, bev_feat):
        bbox = BBox()
        bbox.x, bbox.y, bbox.z, bbox.w, bbox.l, bbox.h, bbox.o = data[:7]
        bbox.z = bbox.z + bbox.h / 2
        bbox.s = score_3d
        bbox.label = label_3d
        bbox.query_feat = query_feat
        bbox.img_feat = img_feat
        bbox.bev_feat = bev_feat
        return bbox

    @classmethod
    def array2bbox(cls, data):
        bbox = BBox()
        bbox.x, bbox.y, bbox.z, bbox.o, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox

    @classmethod
    def dict2bbox(cls, data):
        bbox = BBox()
        bbox.x = data['center_x']
        bbox.y = data['center_y']
        bbox.z = data['center_z']
        bbox.h = data['height']
        bbox.w = data['width']
        bbox.l = data['length']
        bbox.o = data['heading']
        if 'score' in data.keys():
            bbox.s = data['score']
        return bbox

    @classmethod
    def copy_bbox(cls, bboxa, bboxb):
        bboxa.x = bboxb.x
        bboxa.y = bboxb.y
        bboxa.z = bboxb.z
        bboxa.l = bboxb.l
        bboxa.w = bboxb.w
        bboxa.h = bboxb.h
        bboxa.o = bboxb.o
        bboxa.s = bboxb.s
        return

    @classmethod
    def box2corners2d(cls, bbox):
        """ the coordinates for bottom corners
        """
        bottom_center = torch.stack([bbox.x, bbox.y, bbox.z - bbox.h / 2])
        cos, sin = torch.cos(bbox.o), torch.sin(bbox.o)
        pc0 = torch.stack([bbox.x + cos * bbox.l / 2 + sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 - cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc1 = torch.stack([bbox.x + cos * bbox.l / 2 - sin * bbox.w / 2,
                        bbox.y + sin * bbox.l / 2 + cos * bbox.w / 2,
                        bbox.z - bbox.h / 2])
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1

        return [pc0, pc1, pc2, pc3]

    @classmethod
    def box2corners3d(cls, bbox):
        """ the coordinates for bottom corners
        """
        center = torch.stack([bbox.x, bbox.y, bbox.z])
        bottom_corners = torch.stack(BBox.box2corners2d(bbox))
        up_corners = 2 * center - bottom_corners
        corners = torch.cat([up_corners, bottom_corners], dim=0)
        return corners.tolist()

    @classmethod
    def box2corners3d_2points(cls, bbox):
        return

    @classmethod
    def motion2bbox(cls, bbox, motion):
        result = deepcopy(bbox)
        result.x += motion[0]
        result.y += motion[1]
        result.z += motion[2]
        result.o += motion[3]
        return result

    @classmethod
    def set_bbox_size(cls, bbox, size_array):
        result = deepcopy(bbox)
        result.l, result.w, result.h = size_array
        return result

    @classmethod
    def set_bbox_with_states(cls, prev_bbox, state_array):
        prev_array = BBox.bbox2array(prev_bbox)
        prev_array[:4] += state_array[:4]
        prev_array[4:] = state_array[4:]
        bbox = BBox.array2bbox(prev_array)
        return bbox

    @classmethod
    def box_pts2world(cls, ego_matrix, pcs):
        new_pcs = torch.cat((pcs, np.ones(pcs.shape[0])[:, np.newaxis]),
                                 dim=1)
        new_pcs = ego_matrix @ new_pcs.T
        new_pcs = new_pcs.T[:, :3]
        return new_pcs

    @classmethod
    def edge2yaw(cls, center, edge):
        vec = edge - center
        yaw = np.arccos(vec[0] / np.linalg.norm(vec))
        if vec[1] < 0:
            yaw = -yaw
        return yaw

    @classmethod
    def bbox2world(cls, ego_matrix, box):
        # center and corners
        corners = torch.stack(BBox.box2corners2d(box), dim=0)
        center = BBox.bbox2array(box)[:3][np.newaxis, :]
        center = BBox.box_pts2world(ego_matrix, center)[0]
        corners = BBox.box_pts2world(ego_matrix, corners)
        # heading
        edge_mid_point = (corners[0] + corners[1]) / 2
        yaw = BBox.edge2yaw(center[:2], edge_mid_point[:2])

        result = deepcopy(box)
        result.x, result.y, result.z = center
        result.o = yaw
        return result


class Validity:
    TYPES = ['birth', 'alive', 'death']

    def __init__(self):
        return

    @classmethod
    def valid(cls, state_string):
        tokens = state_string.split('_')
        if tokens[0] == 'birth':
            return True
        if len(tokens) < 3:
            return False
        if tokens[0] == 'alive' and int(tokens[1]) == 1:
            return True
        return False

    @classmethod
    def notoutput(cls, state_string):
        tokens = state_string.split('_')
        if len(tokens) < 3:
            return False
        if tokens[0] == 'alive' and int(tokens[1]) != 1:
            return True
        return False

    @classmethod
    def predicted(cls, state_string):
        state, token = state_string.split('_')
        if state not in Validity.TYPES:
            raise ValueError('type name not existed')

        if state == 'alive' and int(token) != 0:
            return True
        return False

    @classmethod
    def modify_string(cls, state_string, mode):
        tokens = state_string.split('_')
        tokens[1] = str(mode)
        return '{:}_{:}_{:}'.format(tokens[0], tokens[1], tokens[2])

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious
