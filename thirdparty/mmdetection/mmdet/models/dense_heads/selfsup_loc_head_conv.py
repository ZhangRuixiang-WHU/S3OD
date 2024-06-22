import torch
import torch.nn as nn
from mmcv.cnn import normal_init, ConvModule, kaiming_init

from ..builder import HEADS


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
class SelfsupLocHeadConv(nn.Module):
    """
    conv-conv-gap-relu-fc-bn-fc-bn
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 with_bias=True,
                 with_last_bn=True,
                 with_last_norm=True):
        super(SelfsupLocHeadConv, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        # self.conv1_0 = ConvModule(256, 256, 1, norm_cfg=dict(type='BN'))
        self.conv1_1 = ConvModule(256, 512, 3, norm_cfg=dict(type='BN'))
        self.conv1_2 = ConvModule(512, 512, 3, norm_cfg=dict(type='BN'))
        self.conv1_3 = ConvModule(512, 1024, 3, norm_cfg=dict(type='BN'))
        # self.conv2_0 = ConvModule(256, 256, 1, norm_cfg=dict(type='BN'))
        self.conv2_1 = ConvModule(256, 512, 3, norm_cfg=dict(type='BN'))
        self.conv2_2 = ConvModule(512, 512, 3, norm_cfg=dict(type='BN'))
        self.conv2_3 = ConvModule(512, 1024, 3, norm_cfg=dict(type='BN'))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = torch.sigmoid
        self.criterion = torch.nn.functional.l1_loss

        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.bn0 = nn.BatchNorm1d(hid_channels)
        self.fc1 = nn.Linear(hid_channels * 2, hid_channels, bias=with_bias)
        self.bn1 = nn.BatchNorm1d(hid_channels)
        self.fc2 = nn.Linear(hid_channels, 1, bias=False)
        self.init_weights()


    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, t_org_feat, t_aug_feat, s_org_feat,s_aug_feat, gt):
        """Forward head.
        """
        assert t_org_feat.shape == s_org_feat.shape
        if t_org_feat.shape[0] == 0:
            t_org_feat = torch.zeros(2,256,7,7,device=t_org_feat.device,dtype=t_org_feat.dtype)
            t_aug_feat = torch.zeros(2,256,7,7,device=t_aug_feat.device,dtype=t_aug_feat.dtype)
            s_org_feat = torch.zeros(2,256,7,7,device=s_org_feat.device,dtype=s_org_feat.dtype)
            s_aug_feat = torch.zeros(2,256,7,7,device=s_aug_feat.device,dtype=s_aug_feat.dtype)
            gt = torch.zeros(2,device=t_org_feat.device,dtype=t_org_feat.dtype)
        if t_org_feat.shape[0] == 1:
            t_org_feat = t_org_feat.repeat(2,1,1,1)
            t_aug_feat = t_aug_feat.repeat(2,1,1,1)
            s_org_feat = s_org_feat.repeat(2,1,1,1)
            s_aug_feat = s_aug_feat.repeat(2,1,1,1)
            gt = gt.repeat(2)
        # t_org_feat = self.conv1_3(self.conv1_2(self.conv1_1(self.conv1_0(t_org_feat))))
        # s_org_feat = self.conv2_3(self.conv2_2(self.conv2_1(self.conv2_0(s_org_feat))))
        t_org_feat = self.conv1_3(self.conv1_2(self.conv1_1(t_org_feat)))
        s_org_feat = self.conv2_3(self.conv2_2(self.conv2_1(s_org_feat)))
        t_f1 = t_org_feat.reshape(-1,self.in_channels)
        t_f1 = self.relu(self.bn0(self.fc0(t_f1)))
        t_f1 = t_f1.reshape(-1, self.hid_channels)
        s_f1 = s_org_feat.reshape(-1,self.in_channels)
        s_f1 = self.relu(self.bn0(self.fc0(s_f1)))
        s_f1 = s_f1.reshape(-1, self.hid_channels)
        
        # t_aug_feat = self.conv1_3(self.conv1_2(self.conv1_1(self.conv1_0(t_aug_feat))))
        # s_aug_feat = self.conv2_3(self.conv2_2(self.conv2_1(self.conv2_0(s_aug_feat))))
        t_aug_feat = self.conv1_3(self.conv1_2(self.conv1_1(t_aug_feat)))
        s_aug_feat = self.conv2_3(self.conv2_2(self.conv2_1(s_aug_feat)))
        t_f2 = t_aug_feat.reshape(-1,self.in_channels)
        t_f2 = self.relu(self.bn0(self.fc0(t_f2)))
        t_f2 = t_f2.reshape(-1, self.hid_channels)
        s_f2 = s_aug_feat.reshape(-1,self.in_channels)
        s_f2 = self.relu(self.bn0(self.fc0(s_f2)))
        s_f2 = s_f2.reshape(-1, self.hid_channels)

        loc_f1 = torch.cat([t_f1, s_f2], dim=-1)
        loc_f2 = torch.cat([s_f1, t_f2], dim=-1)

        mid1 = self.relu(self.bn1(self.fc1(loc_f1)))
        mid2 = self.relu(self.bn1(self.fc1(loc_f2)))

        pred_loc1 = self.sigmoid(self.fc2(mid1).reshape(-1))
        pred_loc2 = self.sigmoid(self.fc2(mid2).reshape(-1))

        if pred_loc1.shape[0] == 0 or pred_loc2.shape[0] == 0:
            return None
        else:
            loss1 = self.criterion(pred_loc1, gt, reduction='mean')
            loss2 = self.criterion(pred_loc2, gt, reduction='mean')
            return (loss1 + loss2) / 2