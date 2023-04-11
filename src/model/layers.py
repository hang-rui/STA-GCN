import torch
from torch import nn
import logging
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
import time
class STA_GC(nn.Module):
    def __init__(self, in_channel, out_channel,  A, kernel_size, stride, depth, t_scale, **kwargs):
        super(STA_GC, self).__init__()
        temporal_window_size, max_graph_distance = kernel_size
        num_T = 288 // t_scale
        self.in_channel = in_channel
        if in_channel < 64: # init block
            self.STAGC = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                SpatialGraphConv(in_channel, out_channel, max_graph_distance, A, **kwargs),
                SpatialAdaptiveGraphConv(out_channel, out_channel,num_T, **kwargs),
                TemporalAdaptiveGraphConv(out_channel, out_channel, num_T, **kwargs),
            )
        else:
            self.STAGC = nn.Sequential(
                SpatialGraphConv(in_channel, out_channel, max_graph_distance, A, **kwargs),
                SpatialAdaptiveGraphConv(out_channel, out_channel,num_T, **kwargs),
                TemporalAdaptiveGraphConv(out_channel, out_channel, num_T, **kwargs),
            )
        self.TCN = nn.Sequential(
            Multi_Scale_Temporal_Layer(out_channel, temporal_window_size, stride=stride, **kwargs),
            Multi_Scale_Temporal_Layer(out_channel, temporal_window_size, stride=1, **kwargs),
        )

        

    def forward(self, x):
        x = self.STAGC(x)
        x = self.TCN(x)
        return x


class Multi_Scale_Temporal_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, stride=1, residual=True, **kwargs):
        super(Multi_Scale_Temporal_Layer, self).__init__()
        dilations = [1, 2, 3, 4]
        padding = [(temporal_window_size + (temporal_window_size-1) * (dilation-1) - 1) // 2 for dilation in dilations]
        num_branches = len(dilations) + 2
        inner_channel = channel // 4
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
                act,
                nn.Conv2d(inner_channel, inner_channel, (temporal_window_size,1), (stride, 1), (padding[i], 0), dilation=(dilations[i], 1), bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
            for i in range(len(dilations))
        ])
        
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
                act,
                nn.MaxPool2d((temporal_window_size, 1), (stride, 1), (padding[0], 0)),
                nn.BatchNorm2d(inner_channel),
        ))

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(inner_channel),
        ))

        self.transform = nn.Sequential(
            nn.BatchNorm2d(inner_channel * num_branches),
            act,
            nn.Conv2d(inner_channel * num_branches, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        x = torch.cat(branch_outs, dim=1)
        x = self.transform(x)
        return x + res

class Zero_Layer(nn.Module):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0


class SGraphGen(nn.Module):
    def __init__(self, in_channel, bias, act, reduct_ratio):
        super(SGraphGen, self).__init__()
        
        squeezed_channel = in_channel // reduct_ratio
        self.phi = nn.Conv2d(in_channel, squeezed_channel, 1, bias=False)
        self.psi = nn.Conv2d(in_channel, squeezed_channel, 1, bias=False)
        
    def forward(self, x):
        n, c, t, v = x.size()
        x1 = self.phi(x)
        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x2 = self.psi(x)
        x2 = x2.permute(0, 2, 1, 3).contiguous()
        x = x1 @ x2
        x = torch.nn.functional.normalize(x, dim=-1)
        return x

class TGraphGen(nn.Module):
    def __init__(self, in_channel, bias, act, reduct_ratio):
        super(TGraphGen, self).__init__()
        
        squeezed_channel = in_channel // reduct_ratio
        self.phi = nn.Conv2d(in_channel, squeezed_channel, 1, bias=False)
        self.psi = nn.Conv2d(in_channel, squeezed_channel, 1, bias=False)
        
    def forward(self, x):
        n, c, t, v = x.size()
        x1 = self.phi(x)
        x1 = x1.permute(0, 3, 2, 1).contiguous()
        x2 = self.psi(x)
        x2 = x2.permute(0, 3, 1, 2).contiguous()
        x = x1 @ x2
        x = torch.nn.functional.normalize(x, dim=-1)
        return x


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, A, bias, act, edge, **kwargs):
        super(SpatialGraphConv, self).__init__()

        self.s_kernel_size = max_graph_distance + 1
        self.A = nn.Parameter(A[:self.s_kernel_size], requires_grad=False)
        self.gcn = nn.Conv2d(in_channel, out_channel*self.s_kernel_size, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1

        if in_channel == out_channel:
            self.residual = nn.Identity()
        elif in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A * self.edge)).contiguous()
        x = self.act(self.bn(x) + res)
        return x


class SpatialAdaptiveGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel,num_T, bias, act, reduct_ratio, **kwargs):
        super(SpatialAdaptiveGraphConv, self).__init__()
        self.act = act
        self.SGraphGen = SGraphGen(in_channel, bias, act, reduct_ratio)
        self.gcn = nn.Conv2d(in_channel, out_channel, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        if in_channel == out_channel:
            self.residual = nn.Identity()
        elif in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        dynamic_A = self.SGraphGen(x)
        res = self.residual(x)
        x = self.gcn(x)
        x = torch.einsum('nctv,ntvw->nctw', (x, dynamic_A)).contiguous()
        x = self.act(self.bn(x) + res)
        return x




class TemporalAdaptiveGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel,num_T, bias, act, reduct_ratio, **kwargs):
        super(TemporalAdaptiveGraphConv, self).__init__()

        self.TGraphGen = TGraphGen(in_channel, bias, act, reduct_ratio)
        self.tgcn = nn.Conv2d(in_channel, out_channel, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act
        if in_channel == out_channel:
            self.residual = nn.Identity()
        elif in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        dynamic_A = self.TGraphGen(x)
        res = self.residual(x)
        x = self.tgcn(x)
        x = torch.einsum('ncqv,nvqt->nctv', (x, dynamic_A)).contiguous()
        x = self.act(self.bn(x) + res)
        return x