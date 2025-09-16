# this is the implementation of the MeidaPipe models:
# https://github.com/positive666/mediapipe_PoseEstimation_pytorch
# to suport the ira fragment, we make the in_channels and the resolution as parameters

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from .blazebase import BlazeLandmark, BlazeBlock

class BlazeHandLandmark(BlazeLandmark):
    """The hand landmark model from MediaPipe.
    
    """
    def __init__(self, resolution=256, in_channels=3):
        super(BlazeHandLandmark, self).__init__()

        # size of ROIs used for input
        self.resolution = resolution
        self.in_channels = in_channels
        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=24, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock(24, 24, 5),
            BlazeBlock(24, 24, 5),
            BlazeBlock(24, 48, 5, 2),
        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 96, 5, 2),
        )

        self.backbone3 = nn.Sequential(
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5, 2),
        )

        self.backbone4 = nn.Sequential(
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5, 2),
        )

        self.blaze5 = BlazeBlock(96, 96, 5)
        self.blaze6 = BlazeBlock(96, 96, 5)
        self.conv7 = nn.Conv2d(96, 48, 1, bias=True)

        self.backbone8 = nn.Sequential(
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 96, 5, 2),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
        )

        self.hand_flag = nn.Conv2d(288, 1, 2, bias=True)
        self.handed = nn.Conv2d(288, 1, 2, bias=True)
        self.landmarks = nn.Conv2d(288, 63, 2, bias=True)


    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0, 21, 3))

        x = F.pad(x, (0, 1, 0, 1), "constant", 0)

        x = self.backbone1(x)
        y = self.backbone2(x)
        z = self.backbone3(y)
        w = self.backbone4(z)

        z = z + F.interpolate(w, scale_factor=2, mode='bilinear')
        z = self.blaze5(z)

        y = y + F.interpolate(z, scale_factor=2, mode='bilinear')
        y = self.blaze6(y)
        y = self.conv7(y)

        x = x + F.interpolate(y, scale_factor=2, mode='bilinear')

        x = self.backbone8(x)

        hand_flag = self.hand_flag(x).view(-1).sigmoid()
        handed = self.handed(x).view(-1).sigmoid()
        landmarks = self.landmarks(x).view(-1, 21, 3) / 256

        return landmarks, hand_flag, handed
        # hand_flag is the probability of the hand in the image, handed is the probability of the hand is left hand, landmarks is the 3d coordinates of the 21 joints of the hand


if __name__ == '__main__':
    model = BlazeHandLandmark(resolution=256, in_channels=1)
    landmarks, hand_flag, handed = model(torch.randn(2, 1, 256, 256))
    print(hand_flag.shape, handed.shape, landmarks.shape)
    # torch.Size([2]) torch.Size([2]) torch.Size([2, 21, 3])
    summary(model, input_size=(2, 1, 256, 256))
    '''
    Total params: 2,013,425
    Trainable params: 2,013,425
    Non-trainable params: 0
    Total mult-adds (M): 786.59
    ==========================================================================================
    Input size (MB): 0.52
    Forward/backward pass size (MB): 119.67
    Params size (MB): 8.05
    Estimated Total Size (MB): 128.25
    '''