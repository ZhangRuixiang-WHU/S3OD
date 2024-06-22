import torch
import torch.nn as nn
from mmcv.cnn import normal_init, ConvModule, kaiming_init

from mmcv.runner import BaseModule
from ..builder import HEADS, build_neck


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@HEADS.register_module
class SelfsupByolHead(BaseModule):
    """
    conv-conv-gap-relu-fc-bn-fc-bn
    """


    def __init__(self,
                 predictor,
                 weight_loss=0.5):
        super(SelfsupByolHead, self).__init__()
        self.predictor = build_neck(predictor)
        self.weight_loss = weight_loss
        # self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = torch.sigmoid
        # self.criterion = torch.nn.functional.l1_loss

        # self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        # self.bn0 = nn.BatchNorm1d(hid_channels)
        # self.fc1 = nn.Linear(hid_channels * 2, hid_channels, bias=with_bias)
        # self.bn1 = nn.BatchNorm1d(hid_channels)
        # self.fc2 = nn.Linear(hid_channels, 1, bias=False)
        # self.init_weights()


    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, t_org_feat, t_aug_feat, s_org_feat,s_aug_feat, gt):
        """Forward head.
        """
        assert t_org_feat.shape == s_org_feat.shape
        t_f1 = t_org_feat#.reshape(-1,256)
        s_f1 = s_org_feat#.reshape(-1,256)
        
        # t_aug_feat = self.conv1_2(self.conv1_1(t_aug_feat))
        # s_aug_feat = self.conv2_2(self.conv2_1(s_aug_feat))
        # t_f2 = t_aug_feat.reshape(-1,self.in_channels)
        # s_f2 = s_aug_feat.reshape(-1,self.in_channels)

        pred_1 = self.predictor([s_f1])[0]
        target_1 = t_f1
        # target_1 = t_f1.detach()
        pred_norm_1 = nn.functional.normalize(pred_1, dim=1)
        target_norm_1 = nn.functional.normalize(target_1, dim=1)
        loss = -(pred_norm_1 * target_norm_1).sum(dim=1).mean()
        return loss * self.weight_loss