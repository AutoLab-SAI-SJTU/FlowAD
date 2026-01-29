# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

@LOSSES.register_module()
class InfoNCELoss(nn.Module):
    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0,
                 temperature=0.5,
                 ):
        super(InfoNCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self,
                embed_predict,
                embed_anchor,
                embed_neg,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        loss = self.loss_weight * self.compute(embed_predict,
                embed_anchor,
                embed_neg,)
        return loss
    
    def compute(self, z_i, z_j, neg_samples):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        neg_samples = F.normalize(neg_samples, dim=2)

        positive_similarity = (z_i * z_j).sum(dim=1, keepdim=True) / self.temperature  # [batch_size, 1]
        negative_similarity = torch.bmm(neg_samples, z_i.unsqueeze(2)).squeeze(2) / self.temperature  # [batch_size, num_neg_samples]

        all_similarities = torch.cat([positive_similarity, negative_similarity], dim=1)  # [batch_size, num_neg_samples + 1]

        numerator = torch.exp(positive_similarity)  # [batch_size, 1]
        denominator = torch.exp(all_similarities).sum(dim=1, keepdim=True)  # [batch_size, num_neg_samples + 1]->[batch_size, 1]
        individual_losses = -torch.log(numerator / denominator).squeeze()
        loss = individual_losses.mean()

        return loss

@LOSSES.register_module()
class ContrastiveLoss(nn.Module):
    def __init__(self,
                 margin=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 ):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.margin = margin

    def forward(self,
                label,
                predict,
                neg,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        loss = self.loss_weight * self.compute(label,
                predict,
                neg,)
        return loss
    
    def compute(self, y, x1, x2):
        dist = F.pairwise_distance(x1, x2)
        loss = y * torch.pow(dist, 2) + (1 - y) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return loss.mean()

@LOSSES.register_module()
class TripletLoss(nn.Module):
    def __init__(self,
                 alpha=0.2,
                 reduction='mean',
                 loss_weight=1.0,
                 ):
        super(TripletLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self,
                anchor, 
                predict, 
                neg,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        loss = self.loss_weight * self.compute(anchor,
                predict,
                neg,)
        return loss
    
    def compute(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(pos_dist - neg_dist + self.alpha, min=0.0)
        
        return loss.mean()


# from mile.constants import SEMANTIC_SEG_WEIGHTS

class SegmentationLoss(nn.Module):
    def __init__(self, use_top_k=False, top_k_ratio=1.0, use_weights=False, poly_one=False, poly_one_coefficient=0.0):
        super().__init__()
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.use_weights = use_weights
        self.poly_one = poly_one
        self.poly_one_coefficient = poly_one_coefficient

        # if self.use_weights:
        #     self.weights = SEMANTIC_SEG_WEIGHTS

    def forward(self, prediction, target):
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)

        # if self.use_weights:
        #     weights = torch.tensor(self.weights, dtype=prediction.dtype, device=prediction.device)
        # else:
        weights = None
        
        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=weights,
        )

        if self.poly_one:
            prob = torch.exp(-loss)
            loss_poly_one = self.poly_one_coefficient * (1-prob)
            loss = loss + loss_poly_one

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss = loss.topk(k, dim=-1)[0]

        return torch.mean(loss)


class RegressionLoss(nn.Module):
    def __init__(self, norm, channel_dim=-1):
        super().__init__()
        self.norm = norm
        self.channel_dim = channel_dim

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=self.channel_dim, keepdims=True)
        return loss.mean()


class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()

        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdims=True)

        return loss[mask].mean()


class ProbabilisticLoss(nn.Module):
    """ Given a prior distribution and a posterior distribution, this module computes KL(posterior, prior)"""
    def __init__(self, remove_first_timestamp=True):
        super().__init__()
        self.remove_first_timestamp = remove_first_timestamp

    def forward(self, prior_mu, prior_sigma, posterior_mu, posterior_sigma):
        posterior_var = posterior_sigma[:, 1:] ** 2
        prior_var = prior_sigma[:, 1:] ** 2

        posterior_log_sigma = torch.log(posterior_sigma[:, 1:])
        prior_log_sigma = torch.log(prior_sigma[:, 1:])

        kl_div = (
                prior_log_sigma - posterior_log_sigma - 0.5
                + (posterior_var + (posterior_mu[:, 1:] - prior_mu[:, 1:]) ** 2) / (2 * prior_var)
        )
        first_kl = - posterior_log_sigma[:, :1] - 0.5 + (posterior_var[:, :1] + posterior_mu[:, :1] ** 2) / 2
        kl_div = torch.cat([first_kl, kl_div], dim=1)

        # Sum across channel dimension
        # Average across batch dimension, keep time dimension for monitoring
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss

@LOSSES.register_module()
class WMLoss(nn.Module):
    def __init__(self, alpha=0.75, loss_weight=1.0):
        super(WMLoss, self).__init__()
        self.alpha = alpha
        # self.loss = ProbabilisticLoss(remove_first_timestamp=True)
        self.loss_weight = loss_weight
    
    def compute(self, prior_mu, prior_sigma, posterior_mu, posterior_sigma):
        posterior_var = posterior_sigma[:, 1:] ** 2
        prior_var = prior_sigma[:, 1:] ** 2

        posterior_log_sigma = torch.log(posterior_sigma[:, 1:])
        prior_log_sigma = torch.log(prior_sigma[:, 1:])

        kl_div = (
                prior_log_sigma - posterior_log_sigma - 0.5
                + (posterior_var + (posterior_mu[:, 1:] - prior_mu[:, 1:]) ** 2) / (2 * prior_var)
        )
        first_kl = - posterior_log_sigma[:, :1] - 0.5 + (posterior_var[:, :1] + posterior_mu[:, :1] ** 2) / 2
        kl_div = torch.cat([first_kl, kl_div], dim=1)

        # Sum across channel dimension
        # Average across batch dimension, keep time dimension for monitoring
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss

    def forward(self, prior, posterior):
        prior_mu, prior_sigma = prior['mu'], prior['sigma']
        posterior_mu, posterior_sigma = posterior['mu'], posterior['sigma']
        prior_loss = self.compute(prior_mu, prior_sigma, posterior_mu.detach(), posterior_sigma.detach())
        posterior_loss = self.compute(prior_mu.detach(), prior_sigma.detach(), posterior_mu, posterior_sigma)

        return self.loss_weight * (self.alpha * prior_loss + (1 - self.alpha) * posterior_loss)
