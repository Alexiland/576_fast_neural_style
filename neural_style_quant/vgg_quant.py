from collections import namedtuple

import torch
from torchvision import models
import sys
import torch.nn as nn

sys.path.append("neual_style_quant")
from modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN

ACT_FW = 0
ACT_BW = 0
GRAD_ACT_ERROR = 0
GRAD_ACT_GC = 0
WEIGHT_BITS = 0
MOMENTUM = 0.9

DWS_BITS = 8
DWS_GRAD_BITS = 16


def Conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1, fix_prec=False):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW,
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC, weight_bits=WEIGHT_BITS,
                   fix_prec=fix_prec)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, fix_prec=False):
    return QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                   padding=padding, dilation=dilation, groups=groups, bias=bias, momentum=MOMENTUM,
                   quant_act_forward=ACT_FW, quant_act_backward=ACT_BW,
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC, weight_bits=WEIGHT_BITS,
                   fix_prec=fix_prec)

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            # print(vgg_pretrained_features[x])
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            # print(vgg_pretrained_features[x])
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            # print(vgg_pretrained_features[x])
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            # print(vgg_pretrained_features[x])
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
