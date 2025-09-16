# the implementation of the cross-keypoints feature fusion and the temporal keypoints feature fusion

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

# the cross-keypoints feature fusion class: 
# the input is the keypoints feature maps with shape (batch_size, 21, w, h)
# we use the 1x1 convolution to exchange the information between the keypoints. hence, the output is the same shape as the input
# And the convolution kernel is initailized as the adjacent matrix of the graph that describe the relationship between the keypoints

class CrossKeypointsFusion(nn.Module):
    def __init__(self, channels, trainable=True, init_adjacent_matrix=False):
        super(CrossKeypointsFusion, self).__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        if init_adjacent_matrix:
            self.conv.weight.data = torch.from_numpy(self.get_adjacent_matrix()).float()
        else:
            self.conv.weight.data = torch.eye(channels).view(channels, channels, 1, 1).float()
        self.conv.weight.requires_grad = trainable

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def get_adjacent_matrix(self):
        '''
        the adjacent matrix of the graph that describe the relationship between the keypoints
        the adjacent matrix is a 21x21 matrix
        the value of the adjacent matrix is 1 if the two keypoints are connected, otherwise the value is 0
        the connections are defined as
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ]
        '''
        adjacent_matrix = np.zeros((self.channels, self.channels))
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ]
        for connection in HAND_CONNECTIONS:
            adjacent_matrix[connection[0], connection[1]] = 1
            adjacent_matrix[connection[1], connection[0]] = 1
        # let the diagonal elements be 1
        for i in range(self.channels):
            adjacent_matrix[i, i] = 1
        # add two more dimensions to the adjacent matrix
        adjacent_matrix = adjacent_matrix.reshape((self.channels, self.channels, 1, 1))
        return adjacent_matrix
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        # x is the keypoints feature maps with shape (batch_size, 21, w, h)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    

# the temporal keypoints feature fusion class:
# the input has two parts: 1) the current keypoints feature with shape (batch_size, 1, 21, w x h)
# 2) the previous keypoints features with shape (batch_size, num_history , 21, w x h)
# We first copy the current keypoints feature and concatenate it with the previous keypoints features
# Then, we use the 1x1 convolution to reduce the channels of the previous keypoints features to 1, to exchange the information between the keypoints
# Finally, we use the element-wise sum to fuse the current keypoints feature and the fused previous keypoints features as a residual connection to the current keypoints feature
# Hence, the output is the shape (batch_size, 1, 21, w x h)
# note: we can have several blocks of the 1x1 conv, batchnorm and relu to reduce the channels of the previous keypoints features

def conv_1x1_bn(inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
        
class TemporalKeypointsFusion(nn.Module):
    def __init__(self, num_history, num_blocks):
        super(TemporalKeypointsFusion, self).__init__()
        self.num_history = num_history
        self.num_blocks = num_blocks
        self.feature_fusion = []
        # generate steps from 1 to num_history
        steps = np.linspace(1, num_history+1, num_blocks, dtype=int)
        steps = steps[::-1]
        in_channel = num_history + 1
        for out_channel in steps:
            self.feature_fusion.append(conv_1x1_bn(in_channel, out_channel))
            in_channel = out_channel
        self.feature_fusion = nn.Sequential(*self.feature_fusion)
        
        
    def forward(self, current_keypoints_feature, previous_keypoints_features):
        # current_keypoints_feature is the current keypoints feature with shape (batch_size, 1, 21, w x h)
        # previous_keypoints_features is the previous keypoints features with shape (batch_size, num_history , 21, w x h)
        all_keypoints_features = torch.cat((current_keypoints_feature, previous_keypoints_features), dim=1)
        fused_keypoints_features = self.feature_fusion(all_keypoints_features)
        fused_keypoints_features = fused_keypoints_features + current_keypoints_feature
        # add a relu layer
        fused_keypoints_features = F.relu(fused_keypoints_features)
        return fused_keypoints_features
    
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


if __name__ == '__main__':
    from torchinfo import summary
    # test the CrossKeypointsFusion
    x = torch.randn(1, 21, 32, 24)
    cross_keypoints_fusion = CrossKeypointsFusion(21, trainable=True, init_adjacent_matrix=False)
    y = cross_keypoints_fusion(x)
    print(y.shape)
    wt = cross_keypoints_fusion.conv.weight.data[:,:,0,0]
    
    print(wt.shape)
    # print(cross_keypoints_fusion.conv.weight.requires_grad)
    summary(cross_keypoints_fusion, input_size=(1, 21, 32, 24))
    
    # test the TemporalKeypointsFusion
    num_history = 5
    num_blocks = 3
    current_keypoints_feature = torch.randn(1, 1, 21, 32*24)
    previous_keypoints_features = torch.randn(1, num_history, 21, 32*24)
    temporal_keypoints_fusion = TemporalKeypointsFusion(num_history, num_blocks)
    fused_keypoints_features = temporal_keypoints_fusion(current_keypoints_feature, previous_keypoints_features)
    print(fused_keypoints_features.shape)
    