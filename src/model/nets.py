import torch
from torch import nn
from .. import utils as U
from .layers import STA_GC
import logging


class STAGCN(nn.Module):
    def __init__(self, data_shape, kernel_size, A, **kwargs):
        super(STAGCN, self).__init__()
        self.A = A
        num_input, num_channel, num_frames, _, _ = data_shape

        self.sta_gcn_networks = nn.Sequential(
            STA_GC(num_channel, 64, A, kernel_size, 1, 1, 1, **kwargs),
            STA_GC(64, 64, A, kernel_size, 1, 1, 1,**kwargs),
            STA_GC(64, 64, A, kernel_size, 1, 1, 1,**kwargs),
            STA_GC(64, 128, A, kernel_size, 2, 1, 1,**kwargs),
            STA_GC(128, 256, A, kernel_size, 2, 1, 2, **kwargs),
        )

        self.classifier = Classifier(256, **kwargs)

        # init parameters
        init_param(self.modules())

    def forward(self, x):
        N, I, C, T, V, M = x.size()
        x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(I, N*M, C, T, V).squeeze()
        x = self.sta_gcn_networks(x)
        # output
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        out = self.classifier(feature).view(N, -1)
        return out, feature

class Classifier(nn.Sequential):
    def __init__(self, curr_channel, num_class, drop_prob, **kwargs):
        super(Classifier, self).__init__()
        self.add_module('gap', nn.AdaptiveAvgPool3d(1))
        self.add_module('dropout', nn.Dropout(drop_prob, inplace=True))
        self.add_module('fc', nn.Conv3d(curr_channel, num_class, kernel_size=1))


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)