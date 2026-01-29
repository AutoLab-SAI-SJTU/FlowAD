""" input form of the data in each frame
"""
from .bbox import BBox

class FrameData:
    def __init__(self, raw_data, ego=None, time_stamp=None, pc=None, det_types=None, aux_info={'is_key_frame':True}):
        boxes_3d = raw_data['boxes_3d']
        scores_3d = raw_data['scores_3d']
        labels_3d = raw_data['labels_3d']
        # track_scores = raw_data['track_scores']
        query_feats = raw_data['query_feats']
        img_feats = raw_data['img_feats']
        bev_feats = raw_data['bev_feats']

        self.dets = []  # detections for each frame
        self.ego = ego  # ego matrix information
        self.pc = pc
        self.det_types = labels_3d  # det_types
        self.time_stamp = time_stamp
        self.aux_info = aux_info

        for i, det in enumerate(boxes_3d):
            self.dets.append(BBox.array2bbox_init(det, scores_3d[i], labels_3d[i], query_feats[i], img_feats[i], bev_feats[i]))

        if not aux_info['is_key_frame']:
            self.dets = [d for d in self.dets if d.s >= 0.5]