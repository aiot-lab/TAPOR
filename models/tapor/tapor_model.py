# we implement the iHand model here with the blocks from FusionBlocks.py, MobileNetEncoder.py, and transformer.py
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import sys

from .FusionBlocks import CrossKeypointsFusion,TemporalKeypointsFusion
from .MobileNetEncoder import MobileEncoder
from .transformer import HandposeEncoder


class Tapor(nn.Module):
    def __init__(self, spatial_encoder_param, 
                 keypoints_encoder_param, 
                 cross_keypoints_fusion_param, 
                 temporal_keypoints_fusion_param,
                 handpose_encoder_param,
                 input_width=32, 
                 input_height=24,
                 batch_size = 24,
                 train=True,
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 ):
        super(Tapor, self).__init__()
        # the spatial_encoder_param is a dictionary that contains the parameters for the spatial encoder
        self.spatial_encoder = MobileEncoder(input_channel = 1, 
                                        last_channel = spatial_encoder_param['last_channel'] , 
                                        width_mult=1., 
                                        interverted_residual_setting = spatial_encoder_param['interverted_residual_setting'], 
                                        upsample_scale_factor = spatial_encoder_param['upsample_scale_factor'],
                                        device = device
                                        ).to(device)
        
        self.keypoints_encoder = MobileEncoder(input_channel = 1,
                                          last_channel = 21 ,
                                          width_mult=1., 
                                          interverted_residual_setting =keypoints_encoder_param['interverted_residual_setting'], 
                                          upsample_scale_factor = keypoints_encoder_param['upsample_scale_factor'],
                                          device = device,
                                          ).to(device)

        self.cross_keypoints_fusion = CrossKeypointsFusion(21, 
                                                           trainable=cross_keypoints_fusion_param['trainable'], 
                                                           init_adjacent_matrix=cross_keypoints_fusion_param['init_adjacent_matrix']
                                                           ).to(device)
        
        self.temporal_keypoints_fusion = TemporalKeypointsFusion(num_history=temporal_keypoints_fusion_param['num_history'],
                                                            num_blocks = temporal_keypoints_fusion_param['num_blocks'],
                                                            ).to(device)
        
        self.num_history = temporal_keypoints_fusion_param['num_history']
        # get the shape of the output of the spatial encoders
        _, sp_feat_c,sp_feat_h,sp_feat_w = self.spatial_encoder.get_output_shape( batch_size = 1, input_channel = 1, h = input_height, w = input_width)
        _, kp_feat_c,kp_feat_h,kp_feat_w = self.keypoints_encoder.get_output_shape( batch_size = 1, input_channel = 1, h = input_height, w = input_width)
        self.sp_c = sp_feat_c
        self.sp_w = sp_feat_w
        self.sp_h = sp_feat_h
        self.kp_feat_w = kp_feat_w
        self.kp_feat_h = kp_feat_h
        self.kp_feat_c = kp_feat_c
        
        self.handpose_encoder = HandposeEncoder(d_model = kp_feat_w * kp_feat_h,
                             kv_dim = sp_feat_c,
                             h = sp_feat_h,
                             w = sp_feat_w,
                             batch = batch_size,
                             c = sp_feat_c,       # use for the position encoding
                             nhead = handpose_encoder_param['num_head'],               # number of heads in the multiheadattention models
                             dim_feedforward = handpose_encoder_param['dim_feedforward'],
                             dropout=0.1,
                             num_layers = handpose_encoder_param['num_layers'],
                             device=device
                             ).to(device)
        # get a decoder with linear layers that input the handpose  features with shape (batch_size, 21, kp_feat_w * kp_feat_h) and output the handpose with shape (batch_size, 21, 3)
        self.decoder = nn.Sequential(
            nn.Linear(kp_feat_w * kp_feat_h, (kp_feat_w * kp_feat_h)//2),
            nn.ReLU(),
            nn.Linear((kp_feat_w * kp_feat_h)//2, 3),
        )
        self.train_model = train

    def forward(self, x):
        if self.train_model:
            # input is a series of thermal maps with shape (batch_size, num_history, h, w)
            # the last frame is the current frame, and the previous frames are the history frames
            # the output is the hand pose of the current frame
            # the output shape is (batch_size, 21, 3)
            bs, num_history, h, w = x.shape
            current_frame = x[:,-1,:,:].unsqueeze(1)
            # all_frame is that of the reshape the input to (batch_size * num_history, 1, h, w)
            all_frame = x.reshape(-1,1,h,w)
            # spatial encoder
            sp_feat = self.spatial_encoder(current_frame) # the output shape is (batch_size, sp_feat_c , sp_feat_h, sp_feat_w)
            # keypoints encoder
            all_kp_feat = self.keypoints_encoder(all_frame)  # output shape is (batch_size * num_history, 21, kp_feat_h, kp_feat_w)
            # go through the cross keypoints fusion
            all_kp_feat = self.cross_keypoints_fusion(all_kp_feat) # output shape is (batch_size * num_history, 21, kp_feat_h, kp_feat_w)
            # reshape the all_kp_feat to (batch_size, num_history, 21, kp_feat_h, kp_feat_w)
            all_kp_feat = all_kp_feat.reshape(bs,num_history,21,self.kp_feat_h,self.kp_feat_w)
            
            if self.num_history == 0:
                current_kp_feat = all_kp_feat.reshape(bs,1,21,-1) # shape is (batch_size, 1, 21, kp_feat_h * kp_feat_w)
                kp_feat = current_kp_feat.reshape(bs,21,-1)
            else:
                current_kp_feat = all_kp_feat[:,-1,:,:,:].unsqueeze(1)
                history_kp_feat = all_kp_feat[:,:-1,:,:,:]
                # view the last two dimensions as a single dimension
                current_kp_feat = current_kp_feat.reshape(bs,1,21,-1) # shape is (batch_size, 1, 21, kp_feat_h * kp_feat_w)
                history_kp_feat = history_kp_feat.reshape(bs,num_history-1,21,-1) # shape is (batch_size, num_history-1, 21,kp_feat_h * kp_feat_w)
                # temporal keypoints fusion
                kp_feat = self.temporal_keypoints_fusion(current_kp_feat, history_kp_feat) # the output shape is (batch_size, 1, 21, kp_feat_h * kp_feat_w)
                # reshape the kp_feat to (batch_size, 21, kp_feat_h * kp_feat_w)
                kp_feat = kp_feat.reshape(bs,21,-1)
            
            # go through the handpose encoder
            handpose_feat,cross_attention_output,cross_attention_map  = self.handpose_encoder(sp_feat, kp_feat)  # the output shape is (21, batch_size, kp_feat_w * kp_feat_h)
            handpose_feat = handpose_feat.permute(1,0,2)
            handpose = self.decoder(handpose_feat)
        else:
            # the input is a tuple with current_frame (shape (batch_size, 1, h, w)) and history keypoint features (batch_size, the number of previous features, 21, h*w)
            # the output is the hand pose of the current frame
            current_frame, history_kp_feat = x
            # spatial encoder
            sp_feat = self.spatial_encoder(current_frame) # the output shape is (batch_size, sp_feat_c , sp_feat_h, sp_feat_w)
            
            # keypoints encoder
            kp_feat = self.keypoints_encoder(current_frame) # output shape is (batch_size, 21, kp_feat_h, kp_feat_w)
            # go through the cross keypoints fusion
            kp_feat = self.cross_keypoints_fusion(kp_feat) # output shape is (batch_size, 21,kp_feat_h, kp_feat_w)
            kp_feat = kp_feat.reshape(kp_feat.shape[0],21,-1)
            kp_feat = kp_feat.unsqueeze(1) # output shape is (batch_size, 1, 21, kp_feat_h * kp_feat_w)
            current_kp_feat = torch.clone(kp_feat)
            
            if self.num_history != 0:
                # temporal keypoints fusion
                kp_feat = self.temporal_keypoints_fusion(kp_feat, history_kp_feat) # the output shape is (batch_size, 1, 21, kp_feat_h, kp_feat_w)
            # reshape the kp_feat to (batch_size, 21, kp_feat_h* kp_feat_w)
            kp_feat = kp_feat.reshape(bs,21,-1)
            # go through the handpose encoder
            handpose_feat,cross_attention_output ,cross_attention_map = self.handpose_encoder(sp_feat, kp_feat)
            handpose_feat = handpose_feat.permute(1,0,2)
            handpose = self.decoder(handpose_feat)
            
        return handpose, current_kp_feat, sp_feat, kp_feat,cross_attention_map


# the tapor teacher model for knowledge distillation
class TaporTeacher(nn.Module):
    def __init__(self, spatial_encoder_param, 
                 keypoints_encoder_param, 
                 cross_keypoints_fusion_param, 
                 temporal_keypoints_fusion_param,
                 handpose_encoder_param,
                 input_width=32, 
                 input_height=24,
                 batch_size = 24,
                 train=True,
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 ):
        super(TaporTeacher, self).__init__()
        # the spatial_encoder_param is a dictionary that contains the parameters for the spatial encoder
        self.spatial_encoder = MobileEncoder(input_channel = 1, 
                                        last_channel = spatial_encoder_param['last_channel'] , 
                                        width_mult=1., 
                                        interverted_residual_setting = spatial_encoder_param['interverted_residual_setting'], 
                                        upsample_scale_factor = spatial_encoder_param['upsample_scale_factor'],
                                        device = device
                                        ).to(device)
        
        self.keypoints_encoder = MobileEncoder(input_channel = 1,
                                          last_channel = 21 ,
                                          width_mult=1., 
                                          interverted_residual_setting =keypoints_encoder_param['interverted_residual_setting'], 
                                          upsample_scale_factor = keypoints_encoder_param['upsample_scale_factor'],
                                          device = device,
                                          ).to(device)

        self.cross_keypoints_fusion = CrossKeypointsFusion(21, 
                                                           trainable=cross_keypoints_fusion_param['trainable'], 
                                                           init_adjacent_matrix=cross_keypoints_fusion_param['init_adjacent_matrix']
                                                           ).to(device)
        
        self.temporal_keypoints_fusion = TemporalKeypointsFusion(num_history=temporal_keypoints_fusion_param['num_history'],
                                                            num_blocks = temporal_keypoints_fusion_param['num_blocks'],
                                                            ).to(device)
        
        self.num_history = temporal_keypoints_fusion_param['num_history']
        # get the shape of the output of the spatial encoders
        _, sp_feat_c,sp_feat_h,sp_feat_w = self.spatial_encoder.get_output_shape( batch_size = 1, input_channel = 1, h = input_height, w = input_width)
        _, kp_feat_c,kp_feat_h,kp_feat_w = self.keypoints_encoder.get_output_shape( batch_size = 1, input_channel = 1, h = input_height, w = input_width)
        self.sp_c = sp_feat_c
        self.sp_w = sp_feat_w
        self.sp_h = sp_feat_h
        self.kp_feat_w = kp_feat_w
        self.kp_feat_h = kp_feat_h
        self.kp_feat_c = kp_feat_c
        
        self.handpose_encoder = HandposeEncoder(d_model = kp_feat_w * kp_feat_h,
                             kv_dim = sp_feat_c,
                             h = sp_feat_h,
                             w = sp_feat_w,
                             batch = batch_size,
                             c = sp_feat_c,       # use for the position encoding
                             nhead = handpose_encoder_param['num_head'],               # number of heads in the multiheadattention models
                             dim_feedforward = handpose_encoder_param['dim_feedforward'],
                             dropout=0.1,
                             num_layers = handpose_encoder_param['num_layers'],
                             device=device
                             ).to(device)
        # get a decoder with linear layers that input the handpose  features with shape (batch_size, 21, kp_feat_w * kp_feat_h) and output the handpose with shape (batch_size, 21, 3)
        self.decoder = nn.Sequential(
            nn.Linear(kp_feat_w * kp_feat_h, (kp_feat_w * kp_feat_h)//2),
            nn.ReLU(),
            nn.Linear((kp_feat_w * kp_feat_h)//2, 3),
        )
        self.train_model = train

    def forward(self, x):
        if self.train_model:
            # input is a series of thermal maps with shape (batch_size, num_history, h, w)
            # the last frame is the current frame, and the previous frames are the history frames
            # the output is the hand pose of the current frame
            # the output shape is (batch_size, 21, 3)
            bs, num_history, h, w = x.shape
            current_frame = x[:,-1,:,:].unsqueeze(1)
            # all_frame is that of the reshape the input to (batch_size * num_history, 1, h, w)
            all_frame = x.reshape(-1,1,h,w)
            # spatial encoder
            sp_feat = self.spatial_encoder(current_frame) # the output shape is (batch_size, sp_feat_c , sp_feat_h, sp_feat_w)
            # keypoints encoder
            all_kp_feat = self.keypoints_encoder(all_frame)  # output shape is (batch_size * num_history, 21, kp_feat_h, kp_feat_w)
            # go through the cross keypoints fusion
            all_kp_feat = self.cross_keypoints_fusion(all_kp_feat) # output shape is (batch_size * num_history, 21, kp_feat_h, kp_feat_w)
            # reshape the all_kp_feat to (batch_size, num_history, 21, kp_feat_h, kp_feat_w)
            all_kp_feat = all_kp_feat.reshape(bs,num_history,21,self.kp_feat_h,self.kp_feat_w)
            
            if self.num_history == 0:
                current_kp_feat = all_kp_feat.reshape(bs,1,21,-1) # shape is (batch_size, 1, 21, kp_feat_h * kp_feat_w)
                kp_feat = current_kp_feat.reshape(bs,21,-1)
            else:
                current_kp_feat = all_kp_feat[:,-1,:,:,:].unsqueeze(1)
                history_kp_feat = all_kp_feat[:,:-1,:,:,:]
                # view the last two dimensions as a single dimension
                current_kp_feat = current_kp_feat.reshape(bs,1,21,-1) # shape is (batch_size, 1, 21, kp_feat_h * kp_feat_w)
                history_kp_feat = history_kp_feat.reshape(bs,num_history-1,21,-1) # shape is (batch_size, num_history-1, 21,kp_feat_h * kp_feat_w)
                # temporal keypoints fusion
                kp_feat = self.temporal_keypoints_fusion(current_kp_feat, history_kp_feat) # the output shape is (batch_size, 1, 21, kp_feat_h * kp_feat_w)
                # reshape the kp_feat to (batch_size, 21, kp_feat_h * kp_feat_w)
                kp_feat = kp_feat.reshape(bs,21,-1)
            
            # go through the handpose encoder
            handpose_feat,cross_attention_output,cross_attention_map  = self.handpose_encoder(sp_feat, kp_feat)  # the output shape is (21, batch_size, kp_feat_w * kp_feat_h)
            handpose_feat = handpose_feat.permute(1,0,2)
            handpose = self.decoder(handpose_feat)
        else:
            # the input is a tuple with current_frame (shape (batch_size, 1, h, w)) and history keypoint features (batch_size, the number of previous features, 21, h*w)
            # the output is the hand pose of the current frame
            current_frame, history_kp_feat = x
            # spatial encoder
            sp_feat = self.spatial_encoder(current_frame) # the output shape is (batch_size, sp_feat_c , sp_feat_h, sp_feat_w)
            
            # keypoints encoder
            kp_feat = self.keypoints_encoder(current_frame) # output shape is (batch_size, 21, kp_feat_h, kp_feat_w)
            # go through the cross keypoints fusion
            kp_feat = self.cross_keypoints_fusion(kp_feat) # output shape is (batch_size, 21,kp_feat_h, kp_feat_w)
            kp_feat = kp_feat.reshape(kp_feat.shape[0],21,-1)
            kp_feat = kp_feat.unsqueeze(1) # output shape is (batch_size, 1, 21, kp_feat_h * kp_feat_w)
            current_kp_feat = torch.clone(kp_feat)
            
            if self.num_history != 0:
                # temporal keypoints fusion
                kp_feat = self.temporal_keypoints_fusion(kp_feat, history_kp_feat) # the output shape is (batch_size, 1, 21, kp_feat_h, kp_feat_w)
            # reshape the kp_feat to (batch_size, 21, kp_feat_h* kp_feat_w)
            kp_feat = kp_feat.reshape(bs,21,-1)
            # go through the handpose encoder
            handpose_feat,cross_attention_output ,cross_attention_map = self.handpose_encoder(sp_feat, kp_feat)
            handpose_feat = handpose_feat.permute(1,0,2)
            handpose = self.decoder(handpose_feat)
            
        return handpose, current_kp_feat, sp_feat, kp_feat,cross_attention_map,cross_attention_output, handpose_feat



if __name__ == "__main__":
    spatial_encoder_param = {
        'last_channel': 128,
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            # [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ],
        'upsample_scale_factor': 10,

    }


    keypoints_encoder_param = {
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 4, 1, 1],
            [6, 8, 2, 1],
            [6, 16, 3, 2],
            [6, 21, 4, 1],
            # [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ],
        'upsample_scale_factor': 1,
    }

    cross_keypoints_fusion_param = {
        'trainable': True,
        'init_adjacent_matrix': True,
    }

    temporal_keypoints_fusion_param ={
        'num_history': 9,
        'num_blocks': 4,
    }

    handpose_encoder_param = {
        'num_head': 4,
        'dim_feedforward': 1024,
        'num_layers': 4,
    }
    
    model = Tapor(spatial_encoder_param, 
                 keypoints_encoder_param, 
                 cross_keypoints_fusion_param, 
                 temporal_keypoints_fusion_param,
                 handpose_encoder_param,
                 input_width=32, 
                 input_height=24,
                 batch_size = 24,
                 train=True)
    # test the forward function
    x = torch.randn(24, 10, 24, 32)
    handpose, current_kp_feat, sp_feat, kp_feat,cross_attention_map = model(x)
    print(handpose.shape)
    print(current_kp_feat.shape)
    print(sp_feat.shape)
    print(kp_feat.shape)
    