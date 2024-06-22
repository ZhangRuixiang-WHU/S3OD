# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss
import numpy as np

from ..builder import ROTATED_LOSSES


def gaussian_dist_pdf(val, mean, var, eps=1e-9):
    simga_constant = 0.3
    return torch.exp(-(val - mean) ** 2.0 / (var + eps) / 2.0) / torch.sqrt(2.0 * np.pi * (var + simga_constant))



@ROTATED_LOSSES.register_module()
class Gaussian1DLoss(nn.Module):
    """Smooth Focal Loss. Implementation of `Circular Smooth Label (CSL).`__

    __ https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40

    Args:
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self, loss_weight=1.0):
        super(Gaussian1DLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        mean_xywha = pred[...,:5]
        sigma_xywha = torch.sigmoid(pred[...,5:])
        gaussian = gaussian_dist_pdf(mean_xywha,target,sigma_xywha)
        loss_box_reg_gaussian = - torch.log(gaussian + 1e-9).mean()/2
        loss_reg = self.loss_weight * loss_box_reg_gaussian

        return loss_reg
