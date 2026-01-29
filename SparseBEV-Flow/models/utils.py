import tempfile
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random

from mmcv.ops.deform_conv import DeformConv2d

def denormalize_pos(normal_pos, x_max, y_max, sigmoid=True):
    max_xy = torch.Tensor([x_max, y_max]).to(normal_pos.device).view(1, 1, 2)
    if sigmoid:
        pos = normal_pos.sigmoid() * max_xy
    else:
        pos = normal_pos * max_xy
    return pos


def normalize_pos(pos, x_max, y_max):
    max_xy = torch.Tensor([x_max, y_max]).to(pos.device).view(1, 1, 2)
    normal_pos = pos / max_xy
    return inverse_sigmoid(normal_pos)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvLN(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel_size=3,  stride=1, padding=1, require_act=True):
        super().__init__()
        if require_act:
            self.module = nn.Sequential(
                nn.Conv2d(input_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                LayerNorm(hidden_channel, data_format="channels_first"),
                nn.ReLU()
            )
        else:
            self.module = nn.Sequential(
                nn.Conv2d(input_channel, hidden_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                LayerNorm(hidden_channel, data_format="channels_first"),
            )

    def forward(self, x):
        # [bs, C, H, W]
        x = self.module(x)
        return x

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DeformConv2dLocal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, deform_groups=1, inner_ker=3):
        super().__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        inner_pad = inner_ker // 2
        self.deform_conv2d = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups, deform_groups=deform_groups,
                                          im2col_step=16)

        self.conv_offset = nn.Conv2d(in_channels, 2*deform_groups*kernel_size*kernel_size, kernel_size=inner_ker, stride=1, padding=inner_pad, groups=groups)
        # init_offset = torch.zeros([2*kernel_size*kernel_size, in_channels, inner_ker, inner_ker], dtype=torch.float32)
        # self.conv_offset.weight = torch.nn.Parameter(init_offset)

        # self.conv_mask = nn.Conv2d(in_channels, kernel_size*kernel_size, kernel_size=inner_ker, stride=1, padding=inner_pad)
        # init_mask = torch.zeros([kernel_size*kernel_size, in_channels, inner_ker, inner_ker], dtype=torch.float32)
        # self.conv_mask.weight = torch.nn.Parameter(init_mask)

    def forward(self, x):
        x = x.contiguous()
        offset = self.conv_offset(x)
        # mask = torch.sigmoid(self.conv_mask(x))
        # print(x.shape, offset.shape)
        out = self.deform_conv2d(x, offset)
        # out = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.conv.weight, mask=mask, padding=(1, 1))
        return out

# General layers
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gumbel_opt_topk(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1, K_ratio=0.7, K=None, opt=torch.sigmoid, return_index=False) -> torch.Tensor:
    # # _gumbels = (-torch.empty_like(
    # #     logits,
    # #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    # #             )  # ~Gumbel(0,1)
    # # more stable https://github.com/pytorch/pytorch/issues/41663
    # gumbel_dist = torch.distributions.gumbel.Gumbel(
    #     torch.tensor(0., device=logits.device, dtype=logits.dtype),
    #     torch.tensor(1., device=logits.device, dtype=logits.dtype))
    # gumbels = gumbel_dist.sample(logits.shape)
    #
    # gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # y_soft = opt(gumbels, dim=dim)
    y_soft = opt(logits, dim=dim) if opt != torch.sigmoid else opt(logits)

    K = int(math.floor(logits.size(dim) * K_ratio)) if K is None else K
    K = 1 if K == 0 else K

    if hard:
        # Straight through.
        # index = y_soft.max(dim, keepdim=True)[1]
        index = torch.topk(y_soft, dim=dim, k=K)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index,
                                                                                                 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        if return_index:
            return ret, index
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class GridMask(nn.Module):
    def __init__(self, ratio=0.5, prob=0.7):
        super(GridMask, self).__init__()
        self.ratio = ratio
        self.prob = prob

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x

        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)

        d = np.random.randint(2, h)
        l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.uint8)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        for i in range(hh // d):
            s = d*i + st_h
            t = min(s + l, hh)
            mask[s:t, :] = 0

        for i in range(ww // d):
            s = d*i + st_w
            t = min(s + l, ww)
            mask[:, s:t] = 0

        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]
        mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
        mask = 1 - mask
        mask = mask.expand_as(x)
        x = x * mask 

        return x.view(n, c, h, w)


def rotation_3d_in_axis(points, angles):
    assert points.shape[-1] == 3
    assert angles.shape[-1] == 1
    angles = angles[..., 0]

    n_points = points.shape[-2]
    input_dims = angles.shape

    if len(input_dims) > 1:
        points = points.reshape(-1, n_points, 3)
        angles = angles.reshape(-1)

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if VERSION.name == 'v0.17.1':
        rot_mat_T = torch.stack([
            rot_cos, -rot_sin, zeros,
            rot_sin, rot_cos, zeros,
            zeros, zeros, ones,
        ]).transpose(0, 1).reshape(-1, 3, 3)
    else:
        rot_mat_T = torch.stack([
            rot_cos, rot_sin, zeros,
            -rot_sin, rot_cos, zeros,
            zeros, zeros, ones,
        ]).transpose(0, 1).reshape(-1, 3, 3)

    points = torch.bmm(points, rot_mat_T)

    if len(input_dims) > 1:
        points = points.reshape(*input_dims, n_points, 3)
    
    return points


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pad_multiple(inputs, img_metas, size_divisor=32):
    _, _, img_h, img_w = inputs.shape

    pad_h = 0 if img_h % size_divisor == 0 else size_divisor - (img_h % size_divisor)
    pad_w = 0 if img_w % size_divisor == 0 else size_divisor - (img_w % size_divisor)

    B = len(img_metas)
    N = len(img_metas[0]['ori_shape'])

    for b in range(B):
        img_metas[b]['img_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]
        img_metas[b]['pad_shape'] = [(img_h + pad_h, img_w + pad_w, 3) for _ in range(N)]

    if pad_h == 0 and pad_w == 0:
        return inputs
    else:
        return F.pad(inputs, [0, pad_w, 0, pad_h], value=0)


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image = image / 255.0

    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0

    h = h * 360.0
    v = v * 255.0

    return torch.stack((h, s, v), dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :] / 360.0
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :] / 255.0

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)
    out = out * 255.0

    return out


class GpuPhotoMetricDistortion:
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

    def __call__(self, imgs):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = imgs[:, [2, 1, 0], :, :]  # BGR to RGB

        contrast_modes = []
        for _ in range(imgs.shape[0]):
            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            contrast_modes.append(random.randint(2))

        for idx in range(imgs.shape[0]):
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                imgs[idx] += delta

            if contrast_modes[idx] == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha

        # convert color from BGR to HSV
        imgs = rgb_to_hsv(imgs)

        for idx in range(imgs.shape[0]):
            # random saturation
            if random.randint(2):
                imgs[idx, 1] *= random.uniform(self.saturation_lower, self.saturation_upper)

            # random hue
            if random.randint(2):
                imgs[idx, 0] += random.uniform(-self.hue_delta, self.hue_delta)

        imgs[:, 0][imgs[:, 0] > 360] -= 360
        imgs[:, 0][imgs[:, 0] < 0] += 360

        # convert color from HSV to BGR
        imgs = hsv_to_rgb(imgs)

        for idx in range(imgs.shape[0]):
            # random contrast
            if contrast_modes[idx] == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    imgs[idx] *= alpha

            # randomly swap channels
            if random.randint(2):
                imgs[idx] = imgs[idx, random.permutation(3)]

        imgs = imgs[:, [2, 1, 0], :, :]  # RGB to BGR

        return imgs


class DumpConfig:
    def __init__(self):
        self.enabled = False
        self.out_dir = './'
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        # self.out_dir = tempfile.mkdtemp()
        self.stage_count = 0
        self.frame_count = 0


DUMP = DumpConfig()


# for backward compatibility
class Version:
    def __init__(self):
        self.name = 'v1.0.0'

VERSION = Version()
