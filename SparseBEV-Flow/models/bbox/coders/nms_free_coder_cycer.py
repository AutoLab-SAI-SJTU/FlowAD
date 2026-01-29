import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from ..utils import denormalize_bbox


@BBOX_CODERS.register_module()
class NMSFreeCoder_Cycer(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """
    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = torch.div(indexs, self.num_classes, rounding_mode='trunc')
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            limit = torch.tensor(self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >= limit[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= limit[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!'
            )

        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))

        return predictions_list
    
    # def decode_train(self, preds_dicts, max_num=300):
    def decode_train(self, all_cls_scores, all_bbox_preds, all_sample_feats,
                     sample_points_cam, sample_points_cam_valid_mask,
                     max_num=300):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        # all_cls_scores = preds_dicts['all_cls_scores'][-1]
        # all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        # all_sample_feats = preds_dicts['all_sample_feats'][-1]
        
        batch_size = all_cls_scores.size()[0]
        cls_list = []
        reg_list = []
        feat_list = []
        sample_points_list = []
        sample_points_mask_list = []
        for i in range(batch_size):
            cls_scores, bbox_preds, sample_feats = all_cls_scores[i], all_bbox_preds[i], all_sample_feats[i]

            # torch.Size([1, 8, 6, 900, 16, 3]) torch.Size([1, 8, 6, 900, 16])
            sample_points_cam_single, sample_points_cam_valid_mask_single = sample_points_cam[i], sample_points_cam_valid_mask[i]

            # max_num = self.max_num

            cls_scores = cls_scores.sigmoid()
            max_num = max_num if max_num < len(cls_scores.view(-1)) else len(cls_scores.view(-1))
            scores, indexs = cls_scores.view(-1).topk(max_num)
            labels = indexs % self.num_classes
            bbox_index = torch.div(indexs, self.num_classes, rounding_mode='trunc')
            bbox_preds = bbox_preds[bbox_index]
            sample_feats = sample_feats[bbox_index]
            sample_points_cam_single = sample_points_cam_single[:,:,bbox_index]
            sample_points_cam_valid_mask_single = sample_points_cam_valid_mask_single[:,:,bbox_index]

            # final_box_preds = denormalize_bbox(bbox_preds)
            final_box_preds = bbox_preds  # has been decoderd in utils
            final_scores = scores 
            final_preds = labels

            # use score threshold
            # if self.score_threshold is not None:
            #     thresh_mask = final_scores > self.score_threshold

            # if self.post_center_range is not None:
            limit = torch.tensor(self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >= limit[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= limit[3:]).all(1)

            # if self.score_threshold:
            #     mask &= thresh_mask
            
            # print('thresh', self.post_center_range, torch.sum(mask), final_box_preds[:10])

            # boxes3d = final_box_preds[mask]
            # scores = final_scores[mask]
            # labels = final_preds[mask]

            bbox_preds = bbox_preds[mask]
            scores = scores[mask]
            sample_feats = sample_feats[mask]
            sample_points_cam_single = sample_points_cam_single[:,:,mask]
            sample_points_cam_valid_mask_single = sample_points_cam_valid_mask_single[:,:,mask]

            cls_list.append(scores)
            reg_list.append(bbox_preds)
            feat_list.append(sample_feats)
            sample_points_list.append(sample_points_cam_single)
            sample_points_mask_list.append(sample_points_cam_valid_mask_single)

        # predictions_dict = {
        #     'bbox_preds': torch.stack(reg_list, dim=0),
        #     'scores': torch.stack(cls_list, dim=0),
        #     'sample_feats': torch.stack(feat_list, dim=0)
        # }
        # return predictions_dict
        return torch.stack(cls_list, dim=0), torch.stack(reg_list, dim=0), torch.stack(feat_list, dim=0), torch.stack(sample_points_list, dim=0), torch.stack(sample_points_mask_list, dim=0)
