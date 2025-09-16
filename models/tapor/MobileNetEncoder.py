# using mobiel-net-like model to process the input with 32x24 resolution. produce 32*k x 24*k resolution output, where k is the scale factor. using deconvolution to upsample the feature map.
# the implementation of the mobilenet V2 is from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
# we builder our encoder based on it
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import sys

import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileEncoder(nn.Module):
    def __init__(self, input_channel = 1, last_channel = 1280 , width_mult=1., interverted_residual_setting =None, upsample_scale_factor = 1, device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") ):
        super(MobileEncoder, self).__init__()
        block = InvertedResidual
        if interverted_residual_setting is None:
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        
        self.device = device

        # building first layer
        # assert input_size % 32 == 0
        self.upsample_scale_factor = upsample_scale_factor
        if upsample_scale_factor == 1:
            pass
        else:
            self.upsample = nn.Upsample(scale_factor=upsample_scale_factor, mode='bilinear', align_corners=True)
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(input_channel, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

    def forward(self, x):
        if self.upsample_scale_factor == 1:
            pass
        else:
            x = self.upsample(x)
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def get_output_shape(self, batch_size = 1, input_channel = 1, h =24, w = 32):
        # run forward once to get the output shape
        # the input shape is (batch_size, input_channel, h, w)
        # the output shape is (batch_size, output_channel, h, w)
        x = torch.randn(batch_size, input_channel, h, w).to(self.device)
        x = self.forward(x)
        return x.shape



if __name__ == '__main__':
    from torchinfo import summary
    # model = MobileNetV2()
    # summary(model, input_size=(1, 3, 32, 24))
    # # print(model)
    
    # model = MobileEncoder(input_channel = 1, last_channel = 1280 , width_mult=1., interverted_residual_setting =None)
    # summary(model, input_size=(1, 1, 32, 24))
    
    interverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        # [6, 96, 3, 1],
        # [6, 160, 3, 2],
        # [6, 320, 1, 1],
    ]
    spatial_encoder = MobileEncoder(input_channel = 1, last_channel = 128 , width_mult=1., interverted_residual_setting =interverted_residual_setting, upsample_scale_factor = 10)
    summary(spatial_encoder, input_size=(1, 1, 32, 24))
    
    interverted_residual_setting = [
        # t, c, n, s
        [1, 4, 1, 1],
        [6, 8, 2, 1],
        [6, 16, 3, 2],
        [6, 21, 4, 1],
        # [6, 96, 3, 1],
        # [6, 160, 3, 2],
        # [6, 320, 1, 1],
    ]
    keypoints_encoder = MobileEncoder(input_channel = 1, last_channel = 21 , width_mult=1., interverted_residual_setting =interverted_residual_setting, upsample_scale_factor = 1)
    summary(keypoints_encoder, input_size=(1, 1, 32, 24))