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
class SelfsupLocHead_1X1(nn.Module):
    """
    conv-conv-gap-relu-fc-bn-fc-bn
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 avg_max,
                 weight_loss,
                 with_bias=True,
                 with_last_bn=True,
                 with_last_norm=True):
        super(SelfsupLocHead_1X1, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.weight_loss = weight_loss
        self.conv1_0 = ConvModule(256, 2048, 1, norm_cfg=dict(type='BN'))
        if avg_max == 'avg':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #AdaptiveMaxPool2d
        else:
            self.avgpool = nn.AdaptiveMaxPool2d((1, 1)) 
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = torch.sigmoid
        self.criterion = torch.nn.functional.l1_loss

        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.bn0 = nn.BatchNorm1d(hid_channels)
        self.fc1 = nn.Linear(hid_channels, out_channels, bias=with_bias)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, 1, bias=False)
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
        t_org_feat = self.conv1_0(t_org_feat)
        s_org_feat = self.conv1_0(s_org_feat)
        t_aug_feat = self.conv1_0(t_aug_feat)
        s_aug_feat = self.conv1_0(s_aug_feat)

        loc_f1 = t_org_feat - s_aug_feat
        loc_f2 = s_org_feat - t_aug_feat
        loc_f1 = self.avgpool(loc_f1)
        loc_f2 = self.avgpool(loc_f2)

        loc_f1 = loc_f1.reshape(-1,self.in_channels)
        loc_f1 = self.relu(self.bn0(self.fc0(loc_f1)))
        loc_f1 = loc_f1.reshape(-1, self.hid_channels)
        loc_f2 = loc_f2.reshape(-1,self.in_channels)
        loc_f2 = self.relu(self.bn0(self.fc0(loc_f2)))
        loc_f2 = loc_f2.reshape(-1, self.hid_channels)

        mid1 = self.relu(self.bn1(self.fc1(loc_f1)))
        mid2 = self.relu(self.bn1(self.fc1(loc_f2)))

        pred_loc1 = self.sigmoid(self.fc2(mid1).reshape(-1))
        pred_loc2 = self.sigmoid(self.fc2(mid2).reshape(-1))

        loss1 = self.criterion(pred_loc1, gt, reduction='mean')
        loss2 = self.criterion(pred_loc2, gt, reduction='mean')
        return (loss1 + loss2) * self.weight_loss