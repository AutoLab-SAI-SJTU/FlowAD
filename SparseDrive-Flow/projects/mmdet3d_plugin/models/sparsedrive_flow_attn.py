from inspect import signature

import torch
from torch import nn
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from .grid_mask import GridMask
from .attention import FlowAttention

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDrive_flow_attn"]


@DETECTORS.register_module()
class SparseDrive_flow_attn(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        embed_dims=256,
        flow_patches=[8,4,2,1],
        flow_ids=[0,1,2,3],
        flow_attn_ids=[2,3],
        flow_pad_single=1,
    ):
        super(SparseDrive_flow_attn, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            ) 
        
        self.flow_patches = flow_patches
        self.flow_ids = flow_ids
        self.flow_attn_ids = flow_attn_ids
        self.flow_pad_single = flow_pad_single
        self.flow_pad_sum = 1 + 2 * flow_pad_single
        self.ids_pad_left_right = torch.tensor([[5,1],[0,2],[1,3],[2,4],[3,5],[4,0]], dtype=torch.long, requires_grad=False)

        self.flow_mlps = nn.ModuleList([
            nn.Linear(self.flow_pad_sum*self.flow_patches[flow_id], self.flow_patches[flow_id]) for flow_id in self.flow_ids
        ])
        self.flow_attns = nn.ModuleList([
            FlowAttention(dim=embed_dims, heads = 4, dim_head = embed_dims//4, dropout = 0.) for flow_attn_id in self.flow_attn_ids
        ])

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )

        # flow operation
        mlvl_feats_flow = []
        T = feature_maps[0].shape[1]//6
        for idx, mlvl_feat in enumerate(feature_maps):
            if idx not in self.flow_ids:
                mlvl_feats_flow.append(mlvl_feat)
                continue
            flow_patch = self.flow_patches[idx]
            B, TN, C, H, W = mlvl_feat.shape
            mlvl_feat_reshaped = mlvl_feat.reshape(B, T, 6, C, H, W)

            feat_pad_left = mlvl_feat_reshaped[:,:,self.ids_pad_left_right[:,0],:,:,-self.flow_pad_single*flow_patch:]
            feat_pad_right = mlvl_feat_reshaped[:,:,self.ids_pad_left_right[:,1],:,:,:self.flow_pad_single*flow_patch]
            mlvl_feat_pad_reshaped = torch.cat([feat_pad_left, mlvl_feat_reshaped, feat_pad_right], dim=-1)

            mlvl_feat_pad_unfold = F.unfold(mlvl_feat_pad_reshaped.flatten(0,2), kernel_size=(H, self.flow_pad_sum*flow_patch), stride=flow_patch)
            mlvl_feat_flow_unfold = mlvl_feat_pad_unfold.reshape(B*TN, C, H, self.flow_pad_sum*flow_patch, W//flow_patch)

            if idx in self.flow_attn_ids:
                mlvl_feat_flow_unfold = rearrange(mlvl_feat_flow_unfold, 'b c h p y -> (b h y) p c')
                mlvl_feat_flow_unfold = self.flow_attns[self.flow_attn_ids.index(idx)](mlvl_feat_flow_unfold)
                mlvl_feat_flow_unfold = rearrange(mlvl_feat_flow_unfold, '(b h y) p c -> b c h p y', b=B*TN, h=H, y=W//flow_patch)

            mlvl_feat_flow_unfold = self.flow_mlps[self.flow_ids.index(idx)](mlvl_feat_flow_unfold.transpose(3,4))
            mlvl_feat_flow_fold = F.fold(mlvl_feat_flow_unfold.transpose(3,4).flatten(1,3), output_size=(H,W), kernel_size=(H, flow_patch), stride=flow_patch)
            mlvl_feats_flow.append(mlvl_feat_flow_fold.reshape(B, TN, C, H, W))
        feature_maps = mlvl_feats_flow
        # raise EOFError


        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
