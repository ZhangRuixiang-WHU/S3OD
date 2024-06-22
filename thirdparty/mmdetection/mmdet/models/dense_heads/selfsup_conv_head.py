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
class SelfsupConvHead(nn.Module):
    """
    conv-conv-conv-relu-fc-bn-fc-bn
    """

    def __init__(self,
                 in_channels,#1024
                 hid_channels,#4096
                 out_channels,#256
                 with_bias=True,
                 with_last_bn=True,
                 with_last_norm=True):
        super(SelfsupConvHead, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.conv1 = ConvModule(256, 512, 3, norm_cfg=dict(type='BN'))
        self.conv2 = ConvModule(512, 512, 3, norm_cfg=dict(type='BN'))
        self.conv3 = ConvModule(512, 1024, 3, norm_cfg=dict(type='BN'))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = torch.sigmoid
        self.criterion = torch.nn.functional.l1_loss
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.bn0 = nn.BatchNorm1d(hid_channels)
        self.fc1 = nn.Linear(hid_channels, out_channels, bias=with_bias)
        # self.bn1 = nn.BatchNorm1d(hid_channels)
        # self.fc2 = nn.Linear(hid_channels, 1, bias=False)
        self.init_weights()


    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        """Forward head.
        """
        if x.shape[0] == 0:
            x = torch.zeros(2,256,7,7,device=x.device,dtype=x.dtype)
        if x.shape[0] == 1:
            x = x.repeat(2,1,1,1)
        x = self.conv3(self.conv2(self.conv1((x))))
        x = x.reshape(-1,self.in_channels)
        x = self.relu(self.bn0(self.fc0(x)))
        x = x.reshape(-1, self.hid_channels)
        x = (self.fc1(x))
        return x