import mmcv
import torch
import numpy as np
from PIL import Image
from numpy import random
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines.transforms_3d import ObjectRangeFilter, ObjectNameFilter
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)

@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]

        results["img"] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i].T for i in
                                range(len(results['extrinsics']))]

        return results

    def _get_rot(self, h):

        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

@PIPELINES.register_module()
class ResizeCropFlipImageMono(ResizeCropFlipImage):
    def __init__(self, with_bbox_2d=False, num_views=6, **kwargs):
        super(ResizeCropFlipImageMono, self).__init__(**kwargs)
        self.with_bbox_2d = with_bbox_2d
        self.num_views = num_views

    def __call__(self, results):
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]

        results["img"] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i].T for i in
                                range(len(results['extrinsics']))]

        # if self.with_bbox_2d:
        #     gt_bboxes_2d = results['gt_bboxes_2d']
        #     gt_labels_2d = results['gt_labels_2d']
        #     gt_bboxes_2d_to_3d = results['gt_bboxes_2d_to_3d']
        #     gt_bboxes_ignore = results['gt_bboxes_ignore']
        #     processed_gt_bboxes_2d = []
        #     processed_gt_labels_2d = []
        #     processed_gt_bboxes_2d_to_3d = []
        #     processed_gt_bboxes_ignore = []
        #     for i in range(min(N, self.num_views)):
        #         bboxes_2d = gt_bboxes_2d[i]
        #         labels_2d = gt_labels_2d[i]
        #         bboxes_2d_to_3d = gt_bboxes_2d_to_3d[i]
        #         bboxes_ignore = gt_bboxes_ignore[i]
        #         # 1. resize
        #         bboxes_2d = bboxes_2d * resize
        #         bboxes_ignore = bboxes_ignore * resize
        #         # 2. crop and filter out-of-image bboxes
        #         bboxes_2d[:, 0::2] = np.clip(bboxes_2d[:, 0::2], crop[0], crop[2])
        #         bboxes_2d[:, 1::2] = np.clip(bboxes_2d[:, 1::2], crop[1], crop[3])
        #         bboxes_2d[:, 0::2] = bboxes_2d[:, 0::2] - crop[0]
        #         bboxes_2d[:, 1::2] = bboxes_2d[:, 1::2] - crop[1]
        #         bboxes_area = (bboxes_2d[:, 2:] - bboxes_2d[:, :2]).prod(1)
        #         valid_mask = bboxes_area > 64
        #         bboxes_2d = bboxes_2d[valid_mask]
        #         labels_2d = labels_2d[valid_mask]
        #         bboxes_2d_to_3d = bboxes_2d_to_3d[valid_mask]

        #         bboxes_ignore[:, 0::2] = np.clip(bboxes_ignore[:, 0::2], crop[0], crop[2])
        #         bboxes_ignore[:, 1::2] = np.clip(bboxes_ignore[:, 1::2], crop[1], crop[3])
        #         bboxes_ignore[:, 0::2] = bboxes_ignore[:, 0::2] - crop[0]
        #         bboxes_ignore[:, 1::2] = bboxes_ignore[:, 1::2] - crop[1]
        #         bboxes_area = (bboxes_ignore[:, 2:] - bboxes_ignore[:, :2]).prod(1)
        #         valid_mask = bboxes_area > 64
        #         bboxes_ignore = bboxes_ignore[valid_mask]
        #         # 3. flip
        #         if flip:
        #             flipped_bboxes = bboxes_2d.copy()
        #             w = crop[2] - crop[0]
        #             flipped_bboxes[..., 0::4] = w - bboxes_2d[..., 2::4]
        #             flipped_bboxes[..., 2::4] = w - bboxes_2d[..., 0::4]
        #             bboxes_2d = flipped_bboxes

        #             flipped_bboxes = bboxes_ignore.copy()
        #             w = crop[2] - crop[0]
        #             flipped_bboxes[..., 0::4] = w - bboxes_ignore[..., 2::4]
        #             flipped_bboxes[..., 2::4] = w - bboxes_ignore[..., 0::4]
        #             bboxes_ignore = flipped_bboxes
        #         # 4. rotate and filter out-of-image bboxes
        #         A = self._get_rot(rotate / 180 * np.pi)
        #         b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        #         b = A.matmul(-b) + b
        #         bbox_corners = np.stack([bboxes_2d[:, 0], bboxes_2d[:, 1], bboxes_2d[:, 0], bboxes_2d[:, 3],
        #                                  bboxes_2d[:, 2], bboxes_2d[:, 3], bboxes_2d[:, 2], bboxes_2d[:, 1]], axis=1).reshape(-1, 4, 2)
        #         bbox_corners = bbox_corners @ A.numpy().T + b.numpy()[None, None]
        #         bboxes_2d = np.concatenate([bbox_corners.min(1), bbox_corners.max(1)], axis=1)
        #         bboxes_2d[:, 0::2] = np.clip(bboxes_2d[:, 0::2], 0, crop[2] - crop[0])
        #         bboxes_2d[:, 1::2] = np.clip(bboxes_2d[:, 1::2], 0, crop[3] - crop[1])
        #         bboxes_area = (bboxes_2d[:, 2:] - bboxes_2d[:, :2]).prod(1)
        #         valid_mask = bboxes_area > 64
        #         bboxes_2d = bboxes_2d[valid_mask]
        #         labels_2d = labels_2d[valid_mask]
        #         bboxes_2d_to_3d = bboxes_2d_to_3d[valid_mask]

        #         bbox_corners = np.stack([bboxes_ignore[:, 0], bboxes_ignore[:, 1], bboxes_ignore[:, 0], bboxes_ignore[:, 3],
        #                                  bboxes_ignore[:, 2], bboxes_ignore[:, 3], bboxes_ignore[:, 2], bboxes_ignore[:, 1]], axis=1).reshape(-1, 4, 2)
        #         bbox_corners = bbox_corners @ A.numpy().T + b.numpy()[None, None]
        #         bboxes_ignore = np.concatenate([bbox_corners.min(1), bbox_corners.max(1)], axis=1)

        #         processed_gt_bboxes_2d.append(bboxes_2d)
        #         processed_gt_labels_2d.append(labels_2d)
        #         processed_gt_bboxes_2d_to_3d.append(bboxes_2d_to_3d)
        #         processed_gt_bboxes_ignore.append(bboxes_ignore)

        #     results['gt_bboxes_2d'] = processed_gt_bboxes_2d
        #     results['gt_labels_2d'] = processed_gt_labels_2d
        #     results['gt_bboxes_2d_to_3d'] = processed_gt_bboxes_2d_to_3d
        #     results['gt_bboxes_ignore'] = processed_gt_bboxes_ignore

        if self.with_bbox_2d:
            gt_bboxes_2d_all = results['gt_bboxes_2d']
            gt_labels_2d_all = results['gt_labels_2d']
            gt_bboxes_2d_to_3d_all = results['gt_bboxes_2d_to_3d']
            gt_bboxes_ignore_all = results['gt_bboxes_ignore']
            processed_gt_bboxes_2d = []
            processed_gt_labels_2d = []
            processed_gt_bboxes_2d_to_3d = []
            processed_gt_bboxes_ignore = []
            for gt_bboxes_2d, gt_labels_2d, gt_bboxes_2d_to_3d, gt_bboxes_ignore in zip(gt_bboxes_2d_all, gt_labels_2d_all, gt_bboxes_2d_to_3d_all, gt_bboxes_ignore_all):
                box_filtered, label_filtered, map_filtered, ignore_filtered = [],[],[],[]
                for i in range(min(N, self.num_views)):
                    bboxes_2d = gt_bboxes_2d[i]
                    labels_2d = gt_labels_2d[i]
                    bboxes_2d_to_3d = gt_bboxes_2d_to_3d[i]
                    bboxes_ignore = gt_bboxes_ignore[i]
                    # 1. resize
                    bboxes_2d = bboxes_2d * resize
                    bboxes_ignore = bboxes_ignore * resize
                    # 2. crop and filter out-of-image bboxes
                    bboxes_2d[:, 0::2] = np.clip(bboxes_2d[:, 0::2], crop[0], crop[2])
                    bboxes_2d[:, 1::2] = np.clip(bboxes_2d[:, 1::2], crop[1], crop[3])
                    bboxes_2d[:, 0::2] = bboxes_2d[:, 0::2] - crop[0]
                    bboxes_2d[:, 1::2] = bboxes_2d[:, 1::2] - crop[1]
                    bboxes_area = (bboxes_2d[:, 2:] - bboxes_2d[:, :2]).prod(1)
                    valid_mask = bboxes_area > 64
                    bboxes_2d = bboxes_2d[valid_mask]
                    labels_2d = labels_2d[valid_mask]
                    bboxes_2d_to_3d = bboxes_2d_to_3d[valid_mask]

                    bboxes_ignore[:, 0::2] = np.clip(bboxes_ignore[:, 0::2], crop[0], crop[2])
                    bboxes_ignore[:, 1::2] = np.clip(bboxes_ignore[:, 1::2], crop[1], crop[3])
                    bboxes_ignore[:, 0::2] = bboxes_ignore[:, 0::2] - crop[0]
                    bboxes_ignore[:, 1::2] = bboxes_ignore[:, 1::2] - crop[1]
                    bboxes_area = (bboxes_ignore[:, 2:] - bboxes_ignore[:, :2]).prod(1)
                    valid_mask = bboxes_area > 64
                    bboxes_ignore = bboxes_ignore[valid_mask]
                    # 3. flip
                    if flip:
                        flipped_bboxes = bboxes_2d.copy()
                        w = crop[2] - crop[0]
                        flipped_bboxes[..., 0::4] = w - bboxes_2d[..., 2::4]
                        flipped_bboxes[..., 2::4] = w - bboxes_2d[..., 0::4]
                        bboxes_2d = flipped_bboxes

                        flipped_bboxes = bboxes_ignore.copy()
                        w = crop[2] - crop[0]
                        flipped_bboxes[..., 0::4] = w - bboxes_ignore[..., 2::4]
                        flipped_bboxes[..., 2::4] = w - bboxes_ignore[..., 0::4]
                        bboxes_ignore = flipped_bboxes
                    # 4. rotate and filter out-of-image bboxes
                    A = self._get_rot(rotate / 180 * np.pi)
                    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
                    b = A.matmul(-b) + b
                    bbox_corners = np.stack([bboxes_2d[:, 0], bboxes_2d[:, 1], bboxes_2d[:, 0], bboxes_2d[:, 3],
                                            bboxes_2d[:, 2], bboxes_2d[:, 3], bboxes_2d[:, 2], bboxes_2d[:, 1]], axis=1).reshape(-1, 4, 2)
                    bbox_corners = bbox_corners @ A.numpy().T + b.numpy()[None, None]
                    bboxes_2d = np.concatenate([bbox_corners.min(1), bbox_corners.max(1)], axis=1)
                    bboxes_2d[:, 0::2] = np.clip(bboxes_2d[:, 0::2], 0, crop[2] - crop[0])
                    bboxes_2d[:, 1::2] = np.clip(bboxes_2d[:, 1::2], 0, crop[3] - crop[1])
                    bboxes_area = (bboxes_2d[:, 2:] - bboxes_2d[:, :2]).prod(1)
                    valid_mask = bboxes_area > 64
                    bboxes_2d = bboxes_2d[valid_mask]
                    labels_2d = labels_2d[valid_mask]
                    bboxes_2d_to_3d = bboxes_2d_to_3d[valid_mask]

                    bbox_corners = np.stack([bboxes_ignore[:, 0], bboxes_ignore[:, 1], bboxes_ignore[:, 0], bboxes_ignore[:, 3],
                                            bboxes_ignore[:, 2], bboxes_ignore[:, 3], bboxes_ignore[:, 2], bboxes_ignore[:, 1]], axis=1).reshape(-1, 4, 2)
                    bbox_corners = bbox_corners @ A.numpy().T + b.numpy()[None, None]
                    bboxes_ignore = np.concatenate([bbox_corners.min(1), bbox_corners.max(1)], axis=1)

                    box_filtered.append(bboxes_2d)
                    label_filtered.append(labels_2d)
                    map_filtered.append(bboxes_2d_to_3d)
                    ignore_filtered.append(bboxes_ignore)

                processed_gt_bboxes_2d.append(box_filtered)
                processed_gt_labels_2d.append(label_filtered)
                processed_gt_bboxes_2d_to_3d.append(map_filtered)
                processed_gt_bboxes_ignore.append(ignore_filtered)

            results['gt_bboxes_2d'] = processed_gt_bboxes_2d
            results['gt_labels_2d'] = processed_gt_labels_2d
            results['gt_bboxes_2d_to_3d'] = processed_gt_bboxes_2d_to_3d
            results['gt_bboxes_ignore'] = processed_gt_bboxes_ignore

        return results

@PIPELINES.register_module()
class ObjectRangeFilterMono(ObjectRangeFilter):
    def __init__(self, with_bbox_2d=False, **kwargs):
        super(ObjectRangeFilterMono, self).__init__(**kwargs)
        self.with_bbox_2d = with_bbox_2d

    def __call__(self, input_dict):
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        mask_numpy = mask.numpy().astype(np.bool)
        gt_labels_3d = gt_labels_3d[mask_numpy]

        # 2d bboxes to 3d bboxes mapping: -1 for not matched to any 3d bbox
        if self.with_bbox_2d:
            gt_ids = np.zeros(len(mask_numpy), dtype=np.int32)
            gt_ids[mask_numpy] = np.arange(len(gt_labels_3d))
            gt_ids[~mask_numpy] = -1
            gt_bboxes_2d_to_3d = input_dict['gt_bboxes_2d_to_3d'][0]
            # assert all([(mapping > -1).all() for mapping in gt_bboxes_2d_to_3d])

            gt_bboxes_2d_to_3d_filtered = []
            for bboxes_2d_to_3d in gt_bboxes_2d_to_3d:
                bboxes_2d_to_3d[bboxes_2d_to_3d > -1] = gt_ids[bboxes_2d_to_3d[bboxes_2d_to_3d > -1]]
                gt_bboxes_2d_to_3d_filtered.append(bboxes_2d_to_3d)
            input_dict['gt_bboxes_2d_to_3d'][0] = gt_bboxes_2d_to_3d_filtered

            # gt_bboxes_2d_to_3d_filtered = []
            # for gt_bboxes_2d_to_3d_single in gt_bboxes_2d_to_3d:
            #     filtered = []
            #     for bboxes_2d_to_3d in gt_bboxes_2d_to_3d_single:
            #         bboxes_2d_to_3d[bboxes_2d_to_3d > -1] = gt_ids[bboxes_2d_to_3d[bboxes_2d_to_3d > -1]]
            #         filtered.append(bboxes_2d_to_3d)
            #     gt_bboxes_2d_to_3d_filtered.append(filtered)
            # input_dict['gt_bboxes_2d_to_3d'] = gt_bboxes_2d_to_3d_filtered

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict


@PIPELINES.register_module()
class ObjectNameFilterMono(ObjectNameFilter):
    def __init__(self, with_bbox_2d=False, **kwargs):
        super(ObjectNameFilterMono, self).__init__(**kwargs)
        self.with_bbox_2d = with_bbox_2d

    def __call__(self, input_dict):
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

        # remove corresponding 2d bboxes
        # if self.with_bbox_2d:
        #     # gt_ids = np.zeros(len(gt_bboxes_mask), dtype=np.int32)
        #     # gt_ids[gt_bboxes_mask] = np.arange(len(input_dict['gt_labels_3d']))
        #     # gt_ids[~gt_bboxes_mask] = -1
        #     # gt_bboxes_2d = input_dict['gt_bboxes_2d']
        #     # gt_labels_2d = input_dict['gt_labels_2d']
        #     # gt_bboxes_2d_to_3d = input_dict['gt_bboxes_2d_to_3d']
        #     # gt_bboxes_2d_filtered = []
        #     # gt_labels_2d_filtered = []
        #     # gt_bboxes_2d_to_3d_filtered = []
        #     # for bboxes_2d, labels_2d, bboxes_2d_to_3d in zip(gt_bboxes_2d, gt_labels_2d, gt_bboxes_2d_to_3d):
        #     #     # 1. filter out 2d bboxes
        #     #     mask_2d = np.array([n in self.labels for n in labels_2d], dtype=np.bool_)
        #     #     gt_bboxes_2d_filtered.append(bboxes_2d[mask_2d])
        #     #     gt_labels_2d_filtered.append(labels_2d[mask_2d])
        #     #     bboxes_2d_to_3d_filtered = bboxes_2d_to_3d[mask_2d]
        #     #     # 2. adjust mapping
        #     #     bboxes_2d_to_3d_filtered[bboxes_2d_to_3d_filtered > -1] = \
        #     #         gt_ids[bboxes_2d_to_3d_filtered[bboxes_2d_to_3d_filtered > -1]]
        #     #     gt_bboxes_2d_to_3d_filtered.append(bboxes_2d_to_3d_filtered)
        #     # input_dict['gt_bboxes_2d'] = gt_bboxes_2d_filtered
        #     # input_dict['gt_labels_2d'] = gt_labels_2d_filtered
        #     # input_dict['gt_bboxes_2d_to_3d'] = gt_bboxes_2d_to_3d_filtered

            
        #     gt_bboxes_2d_all = input_dict['gt_bboxes_2d']
        #     gt_labels_2d_all = input_dict['gt_labels_2d']
        #     gt_bboxes_2d_to_3d_all = input_dict['gt_bboxes_2d_to_3d']
        #     gt_bboxes_2d_filtered = []
        #     gt_labels_2d_filtered = []
        #     gt_bboxes_2d_to_3d_filtered = []
        #     for idx in range(len(gt_labels_2d_all)):
                
        #         gt_bboxes_2d, gt_labels_2d, gt_bboxes_2d_to_3d = gt_bboxes_2d_all[idx], gt_labels_2d_all[idx], gt_bboxes_2d_to_3d_all[idx]
        #         print(gt_labels_2d)
        #         gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_2d],
        #                           dtype=np.bool_)
        #         gt_ids = np.zeros(len(gt_bboxes_mask), dtype=np.int32)
        #         gt_ids[gt_bboxes_mask] = np.arange(len(gt_labels_2d))
        #         gt_ids[~gt_bboxes_mask] = -1
        #         box_filtered = []
        #         label_filtered = []
        #         map_filtered = []
        #         for bboxes_2d, labels_2d, bboxes_2d_to_3d in zip(gt_bboxes_2d, gt_labels_2d, gt_bboxes_2d_to_3d):
        #             # 1. filter out 2d bboxes
        #             mask_2d = np.array([n in self.labels for n in labels_2d], dtype=np.bool_)
        #             box_filtered.append(bboxes_2d[mask_2d])
        #             label_filtered.append(labels_2d[mask_2d])
        #             map_filtered_2d_3d = bboxes_2d_to_3d[mask_2d]
        #             # 2. adjust mapping
        #             map_filtered_2d_3d[map_filtered_2d_3d > -1] = \
        #                 gt_ids[map_filtered_2d_3d[map_filtered_2d_3d > -1]]
        #             map_filtered.append(map_filtered_2d_3d)
        #         gt_bboxes_2d_filtered.append(box_filtered)
        #         gt_labels_2d_filtered.append(label_filtered)
        #         gt_bboxes_2d_to_3d_filtered.append(map_filtered)
        #     input_dict['gt_bboxes_2d'] = gt_bboxes_2d_filtered
        #     input_dict['gt_labels_2d'] = gt_labels_2d_filtered
        #     input_dict['gt_bboxes_2d_to_3d'] = gt_bboxes_2d_to_3d_filtered

        return input_dict

@PIPELINES.register_module()
class ResizeCropFlipImageMono_single(ResizeCropFlipImage):
    def __init__(self, with_bbox_2d=False, num_views=6, **kwargs):
        super(ResizeCropFlipImageMono_single, self).__init__(**kwargs)
        self.with_bbox_2d = with_bbox_2d
        self.num_views = num_views
    
    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(4)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat.numpy()

    def __call__(self, results):
        imgs = results["img"]
        N = len(imgs)
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
       
        if len(results['lidar2img']) == len(results['img']):
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self._img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)
                results['intrinsics'][i] = ida_mat @ results['intrinsics'][i]
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        elif len(results['img']) == 6:
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self._img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)

            for i in range(len(results['lidar2img'])):
                results['intrinsics'][i] = ida_mat @ results['intrinsics'][i]
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        else:
            raise ValueError()
        
        # print((results['intrinsics'][0] @ results['extrinsics'][0].T) == results['lidar2img'][0])
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i].T for i in
                                range(len(results['extrinsics']))]
        # print(len(results['lidar2img']), results['lidar2img'][:7])

        if self.with_bbox_2d:
            gt_bboxes_2d = results['gt_bboxes_2d']
            gt_labels_2d = results['gt_labels_2d']
            gt_bboxes_2d_to_3d = results['gt_bboxes_2d_to_3d']
            gt_bboxes_ignore = results['gt_bboxes_ignore']
            processed_gt_bboxes_2d = []
            processed_gt_labels_2d = []
            processed_gt_bboxes_2d_to_3d = []
            processed_gt_bboxes_ignore = []
            for i in range(min(N, self.num_views)):
                bboxes_2d = gt_bboxes_2d[i]
                labels_2d = gt_labels_2d[i]
                bboxes_2d_to_3d = gt_bboxes_2d_to_3d[i]
                bboxes_ignore = gt_bboxes_ignore[i]
                # 1. resize
                bboxes_2d = bboxes_2d * resize
                bboxes_ignore = bboxes_ignore * resize
                # 2. crop and filter out-of-image bboxes
                bboxes_2d[:, 0::2] = np.clip(bboxes_2d[:, 0::2], crop[0], crop[2])
                bboxes_2d[:, 1::2] = np.clip(bboxes_2d[:, 1::2], crop[1], crop[3])
                bboxes_2d[:, 0::2] = bboxes_2d[:, 0::2] - crop[0]
                bboxes_2d[:, 1::2] = bboxes_2d[:, 1::2] - crop[1]
                bboxes_area = (bboxes_2d[:, 2:] - bboxes_2d[:, :2]).prod(1)
                valid_mask = bboxes_area > 64
                bboxes_2d = bboxes_2d[valid_mask]
                labels_2d = labels_2d[valid_mask]
                bboxes_2d_to_3d = bboxes_2d_to_3d[valid_mask]

                bboxes_ignore[:, 0::2] = np.clip(bboxes_ignore[:, 0::2], crop[0], crop[2])
                bboxes_ignore[:, 1::2] = np.clip(bboxes_ignore[:, 1::2], crop[1], crop[3])
                bboxes_ignore[:, 0::2] = bboxes_ignore[:, 0::2] - crop[0]
                bboxes_ignore[:, 1::2] = bboxes_ignore[:, 1::2] - crop[1]
                bboxes_area = (bboxes_ignore[:, 2:] - bboxes_ignore[:, :2]).prod(1)
                valid_mask = bboxes_area > 64
                bboxes_ignore = bboxes_ignore[valid_mask]
                # 3. flip
                if flip:
                    flipped_bboxes = bboxes_2d.copy()
                    w = crop[2] - crop[0]
                    flipped_bboxes[..., 0::4] = w - bboxes_2d[..., 2::4]
                    flipped_bboxes[..., 2::4] = w - bboxes_2d[..., 0::4]
                    bboxes_2d = flipped_bboxes

                    flipped_bboxes = bboxes_ignore.copy()
                    w = crop[2] - crop[0]
                    flipped_bboxes[..., 0::4] = w - bboxes_ignore[..., 2::4]
                    flipped_bboxes[..., 2::4] = w - bboxes_ignore[..., 0::4]
                    bboxes_ignore = flipped_bboxes
                # 4. rotate and filter out-of-image bboxes
                A = self._get_rot(rotate / 180 * np.pi)
                b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
                b = A.matmul(-b) + b
                bbox_corners = np.stack([bboxes_2d[:, 0], bboxes_2d[:, 1], bboxes_2d[:, 0], bboxes_2d[:, 3],
                                         bboxes_2d[:, 2], bboxes_2d[:, 3], bboxes_2d[:, 2], bboxes_2d[:, 1]], axis=1).reshape(-1, 4, 2)
                bbox_corners = bbox_corners @ A.numpy().T + b.numpy()[None, None]
                bboxes_2d = np.concatenate([bbox_corners.min(1), bbox_corners.max(1)], axis=1)
                bboxes_2d[:, 0::2] = np.clip(bboxes_2d[:, 0::2], 0, crop[2] - crop[0])
                bboxes_2d[:, 1::2] = np.clip(bboxes_2d[:, 1::2], 0, crop[3] - crop[1])
                bboxes_area = (bboxes_2d[:, 2:] - bboxes_2d[:, :2]).prod(1)
                valid_mask = bboxes_area > 64
                bboxes_2d = bboxes_2d[valid_mask]
                labels_2d = labels_2d[valid_mask]
                bboxes_2d_to_3d = bboxes_2d_to_3d[valid_mask]

                bbox_corners = np.stack([bboxes_ignore[:, 0], bboxes_ignore[:, 1], bboxes_ignore[:, 0], bboxes_ignore[:, 3],
                                         bboxes_ignore[:, 2], bboxes_ignore[:, 3], bboxes_ignore[:, 2], bboxes_ignore[:, 1]], axis=1).reshape(-1, 4, 2)
                bbox_corners = bbox_corners @ A.numpy().T + b.numpy()[None, None]
                bboxes_ignore = np.concatenate([bbox_corners.min(1), bbox_corners.max(1)], axis=1)

                processed_gt_bboxes_2d.append(bboxes_2d)
                processed_gt_labels_2d.append(labels_2d)
                processed_gt_bboxes_2d_to_3d.append(bboxes_2d_to_3d)
                processed_gt_bboxes_ignore.append(bboxes_ignore)

            results['gt_bboxes_2d'] = processed_gt_bboxes_2d
            results['gt_labels_2d'] = processed_gt_labels_2d
            results['gt_bboxes_2d_to_3d'] = processed_gt_bboxes_2d_to_3d
            results['gt_bboxes_ignore'] = processed_gt_bboxes_ignore
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['pad_shape'] = [img.shape for img in results['img']]
        

        return results

@PIPELINES.register_module()
class ObjectRangeFilterMono_single(ObjectRangeFilter):
    def __init__(self, with_bbox_2d=False, **kwargs):
        super(ObjectRangeFilterMono_single, self).__init__(**kwargs)
        self.with_bbox_2d = with_bbox_2d

    def __call__(self, input_dict):
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        mask_numpy = mask.numpy().astype(np.bool)
        gt_labels_3d = gt_labels_3d[mask_numpy]

        # 2d bboxes to 3d bboxes mapping: -1 for not matched to any 3d bbox
        if self.with_bbox_2d:
            gt_ids = np.zeros(len(mask_numpy), dtype=np.int32)
            gt_ids[mask_numpy] = np.arange(len(gt_labels_3d))
            gt_ids[~mask_numpy] = -1
            gt_bboxes_2d_to_3d = input_dict['gt_bboxes_2d_to_3d']
            # assert all([(mapping > -1).all() for mapping in gt_bboxes_2d_to_3d])

            gt_bboxes_2d_to_3d_filtered = []
            for bboxes_2d_to_3d in gt_bboxes_2d_to_3d:
                bboxes_2d_to_3d[bboxes_2d_to_3d > -1] = gt_ids[bboxes_2d_to_3d[bboxes_2d_to_3d > -1]]
                gt_bboxes_2d_to_3d_filtered.append(bboxes_2d_to_3d)
            input_dict['gt_bboxes_2d_to_3d'] = gt_bboxes_2d_to_3d_filtered

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        if 'vis_level' in input_dict.keys():
            input_dict['vis_level'] = input_dict['vis_level'][mask_numpy]

        return input_dict


@PIPELINES.register_module()
class ObjectNameFilterMono_single(ObjectNameFilter):
    def __init__(self, with_bbox_2d=False, **kwargs):
        super(ObjectNameFilterMono_single, self).__init__(**kwargs)
        self.with_bbox_2d = with_bbox_2d

    def __call__(self, input_dict):
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]
        if 'vis_level' in input_dict.keys():
            input_dict['vis_level'] = input_dict['vis_level'][gt_bboxes_mask]

        # remove corresponding 2d bboxes
        if self.with_bbox_2d:
            gt_ids = np.zeros(len(gt_bboxes_mask), dtype=np.int32)
            gt_ids[gt_bboxes_mask] = np.arange(len(input_dict['gt_labels_3d']))
            gt_ids[~gt_bboxes_mask] = -1
            gt_bboxes_2d = input_dict['gt_bboxes_2d']
            gt_labels_2d = input_dict['gt_labels_2d']
            gt_bboxes_2d_to_3d = input_dict['gt_bboxes_2d_to_3d']
            gt_bboxes_2d_filtered = []
            gt_labels_2d_filtered = []
            gt_bboxes_2d_to_3d_filtered = []
            for bboxes_2d, labels_2d, bboxes_2d_to_3d in zip(gt_bboxes_2d, gt_labels_2d, gt_bboxes_2d_to_3d):
                # 1. filter out 2d bboxes
                mask_2d = np.array([n in self.labels for n in labels_2d], dtype=np.bool_)
                gt_bboxes_2d_filtered.append(bboxes_2d[mask_2d])
                gt_labels_2d_filtered.append(labels_2d[mask_2d])
                bboxes_2d_to_3d_filtered = bboxes_2d_to_3d[mask_2d]
                # 2. adjust mapping
                bboxes_2d_to_3d_filtered[bboxes_2d_to_3d_filtered > -1] = \
                    gt_ids[bboxes_2d_to_3d_filtered[bboxes_2d_to_3d_filtered > -1]]
                gt_bboxes_2d_to_3d_filtered.append(bboxes_2d_to_3d_filtered)
            input_dict['gt_bboxes_2d'] = gt_bboxes_2d_filtered
            input_dict['gt_labels_2d'] = gt_labels_2d_filtered
            input_dict['gt_bboxes_2d_to_3d'] = gt_bboxes_2d_to_3d_filtered

        return input_dict

@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, img):
        if self.size_divisor is not None:
            pad_h = int(np.ceil(img.shape[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(img.shape[1] / self.size_divisor)) * self.size_divisor
        else:
            pad_h, pad_w = self.size

        pad_width = ((0, pad_h - img.shape[0]), (0, pad_w - img.shape[1]), (0, 0))
        img = np.pad(img, pad_width, constant_values=self.pad_val)
        return img

    def _pad_imgs(self, results):
        padded_img = [self._pad_img(img) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_imgs(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1)
        self.std = 1 / np.array(std, dtype=np.float32).reshape(-1)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        normalized_imgs = []

        for img in results['img']:
            img = img.astype(np.float32)
            if self.to_rgb:
                img = img[..., ::-1]
            img = img - self.mean
            img = img * self.std
            normalized_imgs.append(img)

        results['img'] = normalized_imgs
        results['img_norm_cfg'] = dict(
            mean=self.mean,
            std=self.std,
            to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            ori_dtype = img.dtype
            img = img.astype(np.float32)

            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]

            new_imgs.append(img.astype(ori_dtype))

        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class RandomTransformImage(object):
    def __init__(self, ida_aug_conf=None, training=True):
        self.ida_aug_conf = ida_aug_conf
        self.training = training

    def __call__(self, results):
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        # print(len(results['lidar2img']), results['lidar2img'][:7])
        # print(len(results['lidar2img']), len(results['img']))
        
        if len(results['lidar2img']) == len(results['img']):
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        elif len(results['img']) == 6:
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)

            for i in range(len(results['lidar2img'])):
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        else:
            raise ValueError()

        # print(len(results['lidar2img']), results['lidar2img'][:7])
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['pad_shape'] = [img.shape for img in results['img']]

        return results

    def img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L48
        """
        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])

        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        
        A = get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b

        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

        ida_mat = torch.eye(4)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran

        return img, ida_mat.numpy()

    def sample_augmentation(self):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L247
        """
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']

        if self.training:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    def __init__(self,
                 rot_range=[-0.3925, 0.3925],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0]):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

    def __call__(self, results):
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)
        self.rotate_z(results, rot_angle)
        results["gt_bboxes_3d"].rotate(np.array(rot_angle))

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        # TODO: support translation

        return results

    def rotate_z(self, results, rot_angle):
        rot_cos = torch.cos(torch.tensor(rot_angle))
        rot_sin = torch.sin(torch.tensor(rot_angle))

        rot_mat = torch.tensor([
            [rot_cos, -rot_sin, 0, 0],
            [rot_sin, rot_cos, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        rot_mat_inv = torch.inverse(rot_mat)

        for view in range(len(results['lidar2img'])):
            results['lidar2img'][view] = (torch.tensor(results['lidar2img'][view]).float() @ rot_mat_inv).numpy()

    def scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ])
        scale_mat_inv = torch.inverse(scale_mat)

        for view in range(len(results['lidar2img'])):
            results['lidar2img'][view] = (torch.tensor(results['lidar2img'][view]).float() @ scale_mat_inv).numpy()
