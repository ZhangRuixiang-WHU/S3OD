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
class SelfsupLocHead(nn.Module):
    """
    conv-conv-gap-relu-fc-bn-fc-bn
    """

    def __init__(self,
                #  in_channels,#256*2
                 hid_channels,#256
                 weight_loss = 0.5,
                 with_bias=True,
                 with_last_bn=True,
                 with_last_norm=True):
        super(SelfsupLocHead, self).__init__()
        # self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = torch.sigmoid
        self.criterion = torch.nn.functional.l1_loss
        self.fc1 = nn.Linear(hid_channels*2, hid_channels, bias=with_bias)
        self.bn1 = nn.BatchNorm1d(hid_channels)
        self.fc2 = nn.Linear(hid_channels, 1, bias=False)
        self.init_weights()
        self.weight_loss = weight_loss


    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, t_org_feat, t_aug_feat, s_org_feat,s_aug_feat, gt):
        """Forward head.
        """
        assert t_org_feat.shape == s_org_feat.shape
        t_f1 = t_org_feat.reshape(-1,self.hid_channels)
        s_f1 = s_org_feat.reshape(-1,self.hid_channels)
        t_f2 = t_aug_feat.reshape(-1,self.hid_channels)
        s_f2 = s_aug_feat.reshape(-1,self.hid_channels)

        loc_f1 = torch.cat([t_f1, s_f2], dim=-1)
        loc_f2 = torch.cat([s_f1, t_f2], dim=-1)

        mid1 = self.relu(self.bn1(self.fc1(loc_f1)))
        mid2 = self.relu(self.bn1(self.fc1(loc_f2)))

        pred_loc1 = self.sigmoid(self.fc2(mid1).reshape(-1))
        pred_loc2 = self.sigmoid(self.fc2(mid2).reshape(-1))
        try:
            loss1 = self.criterion(pred_loc1, gt, reduction='mean')
        except:
            print(gt)
        # loss1 = self.criterion(pred_loc1, gt, reduction='mean')
        loss2 = self.criterion(pred_loc2, gt, reduction='mean')
        return (loss1 + loss2) * self.weight_loss