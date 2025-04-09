import cv2
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset,random_split
import os
import argparse
import torch
from tqdm import tqdm
import pickle
from dataset import iHand_dataset, RandomOrthogonalRotation
from models import BlazeHandLandmark, ColorHandPose3D, Mano_MobileNetV2, Tapor
from tapor_model_config import small_setting, base1_setting, base2_setting, large1_setting, large2_setting, base1_setting_varaint1, base1_setting_varaint2
from utils import calculate_mpjpe, calculate_pck, calculate_shift_error
import matplotlib.pyplot as plt 

def inference(model_name,model,data_loader,device):
    model.eval()  # Set the model to evaluation mode
    if model_name == 'baseline3d':
        predicts_all = {
            'joints_3d': [],
            'keypoints_scoremap': [],
        }
        labels_all = {
            'thermal_map': [],
            'joints_3d': [],
            'l_3d_flag': [],
            'keypoints_scoremap': [],
            'joints_2d': [],
            'l_2d_flag': [],
            'l_hand_depth': [],
            'l_left_right_flag': [],
            'ambient_temperature': [],
        }
    elif model_name == 'mediapipe':
        predicts_all = {
            'joints_3d': [],
            'joints_3d_flag': [],
        }
        labels_all = {
            'thermal_map': [],
            'joints_3d': [],
            'l_3d_flag': [],
            'joints_2d': [],
            'l_2d_flag': [],
            'l_hand_depth': [],
            'l_left_right_flag': [],
            'ambient_temperature': [],
        }
    elif model_name == 'mano':
        predicts_all = {
            'joints_3d': [],
            'joints_3d_flag': [],
        }
        labels_all = {
            'thermal_map': [],
            'joints_3d': [],
            'l_3d_flag': [],
            'joints_2d': [],
            'l_2d_flag': [],
            'l_hand_depth': [],
            'l_left_right_flag': [],
            'ambient_temperature': [],
        }
    elif model_name == 'tapor':
        predicts_all = {
            'joints_3d': [],
            'current_kp_feat': [],
            'sp_feat': [],
            'kp_feat': [],
            'attention_map': [],
        }
        labels_all = {
            'thermal_map': [],
            'joints_3d': [],
            'l_3d_flag': [],
            'joints_2d': [],
            'l_2d_flag': [],
            'l_hand_depth': [],
            'l_left_right_flag': [],
            'ambient_temperature': [],
        }
    with torch.no_grad():  # Disable gradient computation during evaluation
        for i, input in enumerate(tqdm(data_loader)):
            thermal_map, ambient_temperature, l_depth_map, l_2d_joint, l_2d_flag, l_3d_joint, l_3d_flag, l_hand_depth, l_left_right_flag, heatmap_label = input
            l_2d_flag = l_2d_flag.float().to(device)  # (batch_size, 1)
            l_2d_joint = l_2d_joint.squeeze().float().to(device) # (batch_size,c ,21, 3), the last dim is unuseful; if c=1, then  (batch_size ,21, 3)
            l_3d_flag = l_3d_flag.float().to(device)  # (batch_size, 1)
            l_3d_joint = l_3d_joint.squeeze().float().to(device) # (batch_size, c, 21, 3); if c=1, then  (batch_size ,21, 3)
            thermal_map = thermal_map.float().to(device)  # (batch_size, c, 256, 256)
            if model_name == 'baseline3d':
                # the channle num: c = 1
                l_left_right_flag = l_left_right_flag.squeeze().float().to(device)  # (batch_size, 2)
                coord_can, keypoints_scoremap = model(thermal_map,l_left_right_flag) # (batch_size, 21, 3), (batch_size, 21, 256, 256)
                # b,c,h,w = keypoints_scoremap.shape
                # heatmap_label = generate_heatmaps(l_2d_joint.cpu().numpy()[:,:,:2], heatmap_size=(h,w), sigma=5)
                predicts_all['joints_3d'].append(coord_can.cpu().numpy())
                predicts_all['keypoints_scoremap'].append(keypoints_scoremap.cpu().numpy())
                labels_all['thermal_map'].append(thermal_map.cpu().numpy())
                labels_all['joints_3d'].append(l_3d_joint.cpu().numpy())
                labels_all['l_3d_flag'].append(l_3d_flag.cpu().numpy())
                try:
                    labels_all['keypoints_scoremap'].append(heatmap_label.cpu().numpy())
                except:
                    labels_all['keypoints_scoremap'].append(heatmap_label)
                labels_all['joints_2d'].append(l_2d_joint.cpu().numpy())
                labels_all['l_2d_flag'].append(l_2d_flag.cpu().numpy())
                labels_all['l_hand_depth'].append(l_hand_depth.numpy())
                labels_all['l_left_right_flag'].append(l_left_right_flag.cpu().numpy())
                labels_all['ambient_temperature'].append(ambient_temperature.numpy())
            elif model_name == 'mano':
                joints_left, joints_right = model(thermal_map) # (batch_size, 21, 3), (batch_size, 21, 256, 256)
                l_left_right_flag = l_left_right_flag.squeeze().float().to(device)
                joints_left = l_left_right_flag[:, 0].unsqueeze(1).unsqueeze(2) * joints_left
                joints_right = l_left_right_flag[:, 1].unsqueeze(1).unsqueeze(2) * joints_right
                landmarks = joints_left + joints_right
                predicts_all['joints_3d'].append(landmarks.cpu().numpy())
                predicts_all['joints_3d_flag'].append(l_3d_flag.cpu().numpy())
                labels_all['thermal_map'].append(thermal_map.cpu().numpy())
                labels_all['joints_3d'].append(l_3d_joint.cpu().numpy())
                labels_all['l_3d_flag'].append(l_3d_flag.cpu().numpy())
                labels_all['joints_2d'].append(l_2d_joint.cpu().numpy())
                labels_all['l_2d_flag'].append(l_2d_flag.cpu().numpy())
                labels_all['l_hand_depth'].append(l_hand_depth.numpy())
                labels_all['l_left_right_flag'].append(l_left_right_flag.cpu().numpy())
                labels_all['ambient_temperature'].append(ambient_temperature.numpy())
            elif model_name == 'mediapipe':
                # the channle num: c = 1
                landmarks, hand_flag, handed = model(thermal_map) # (batch_size, 21, 3), (batch_size), (batch_size)
                predicts_all['joints_3d'].append(landmarks.cpu().numpy())
                predicts_all['joints_3d_flag'].append(hand_flag.cpu().numpy())
                labels_all['thermal_map'].append(thermal_map.cpu().numpy())
                labels_all['joints_3d'].append(l_3d_joint.cpu().numpy())
                labels_all['l_3d_flag'].append(l_3d_flag.cpu().numpy())
                labels_all['joints_2d'].append(l_2d_joint.cpu().numpy())
                labels_all['l_2d_flag'].append(l_2d_flag.cpu().numpy())
                labels_all['l_hand_depth'].append(l_hand_depth.numpy())
                labels_all['l_left_right_flag'].append(l_left_right_flag.numpy())
                labels_all['ambient_temperature'].append(ambient_temperature.numpy())
            elif model_name == 'tapor':
                landmarks, current_kp_feat, sp_feat, kp_feat, attention_map = model(thermal_map)
                # print(landmarks.shape)
                predicts_all['joints_3d'].append(landmarks.cpu().numpy())
                predicts_all['current_kp_feat'].append(current_kp_feat.cpu().numpy())
                predicts_all['sp_feat'].append(sp_feat.cpu().numpy())
                predicts_all['kp_feat'].append(kp_feat.cpu().numpy())
                predicts_all['attention_map'].append(attention_map.cpu().numpy())   
                labels_all['thermal_map'].append(thermal_map.cpu().numpy())
                if l_3d_joint.shape[1] == 21:
                    pass
                else:
                    l_3d_joint = l_3d_joint[:, -1, :, :]  # (batch_size, 21, 3)
                print(l_3d_joint.shape)
                labels_all['joints_3d'].append(l_3d_joint.cpu().numpy())
                labels_all['l_3d_flag'].append(l_3d_flag.cpu().numpy())
                labels_all['joints_2d'].append(l_2d_joint.cpu().numpy())
                labels_all['l_2d_flag'].append(l_2d_flag.cpu().numpy())
                labels_all['l_hand_depth'].append(l_hand_depth.numpy())
                labels_all['l_left_right_flag'].append(l_left_right_flag.numpy())
                labels_all['ambient_temperature'].append(ambient_temperature.numpy())
    return predicts_all,labels_all


def draw_3D_landmarks2(thermal_map,prediction,groundtruth =None):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    height_default, dpi_default= (4.8, 300)
    # height,wide = thermal_map.shape
    height,wide = (256*3,256*3)

    lines = [[0,1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20], [0,5,9,13,17,0]]
    for line in lines:
        x,y,z = [],[],[]
        label_x,label_y,label_z = [],[],[]
        for id in line:
            x.append(-prediction[id][0])
            y.append(-prediction[id][1])
            z.append(prediction[id][2])
            if groundtruth is not None:
                label_x.append(-groundtruth[id][0])
                label_y.append(-groundtruth[id][1])
                label_z.append(groundtruth[id][2])
        ax.plot(x,z,y,color='g', linestyle="-", linewidth=1)
        ax.scatter(x, z, y, s=10, c='r', marker='o')
        if groundtruth is not None:
            ax.plot(label_x,label_z,label_y,color='k', linestyle="-", linewidth=1)
            ax.scatter(label_x, label_z, label_y, s=10, c='b', marker='o')

    # set the limits of the plot to the limits of the data
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(0, 0.6)
    ax.set_zlim(-0.2, 0.2)
    # set the stride of the ticks
    ax.xaxis.set_ticks(np.arange(-0.2, 0.21, 0.1))
    ax.yaxis.set_ticks(np.arange(0, 0.61, 0.1))
    ax.zaxis.set_ticks(np.arange(-0.2, 0.21, 0.1))
    
    fig.canvas.draw()
    fig_str = fig.canvas.tostring_rgb()
    
    plot_img = np.frombuffer(fig_str, dtype=np.uint8).reshape((int(height_default*dpi_default), -1, 3)) #/ 255.0
    plot_img = cv2.resize(plot_img,(wide*3,height*3))
    plt.close(fig)
    return plot_img


def visualization_results_video(log_folder, log_file_name,input_thermal_frames, groundtruth_3d_joints, estimated_3d_joints, t_size = (256*3,256*3),file_names_for_visualization = None):
    # input_thermal_frames: (n, 256, 256, 3)
    # groundtruth_3d_joints: (n, 21, 3)
    # estimated_3d_joints: (n, 21, 3)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    height,wide = t_size
    out = cv2.VideoWriter(log_folder + log_file_name +'_'+"inference.mp4",fourcc,15,(wide*2,height*1))

    n = input_thermal_frames.shape[0]
    for i in tqdm(range(n)):
        frame = input_thermal_frames[i, 0, :,:]
        # apply colormap on thermal image
        frame_norm = (frame - np.min(frame))/ (np.max(frame) - np.min(frame)) * 255
        frame_colormap = cv2.applyColorMap((frame_norm).astype(np.uint8), cv2.COLORMAP_JET) 
        groundtruth = groundtruth_3d_joints[i]
        prediction = estimated_3d_joints[i]
        vis_figure = draw_3D_landmarks2(frame,prediction,groundtruth)
        frame_colormap = cv2.resize(frame_colormap,t_size, interpolation=cv2.INTER_NEAREST)
        vis_figure = cv2.resize(vis_figure,t_size)
        concatenated_image = np.concatenate((frame_colormap,vis_figure),axis=1)
        if file_names_for_visualization is not None:
            cv2.putText(concatenated_image, file_names_for_visualization[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(concatenated_image)
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training and Testing')
    parser.add_argument('-m', '--model_name', type=str, default='baseline3d', help='the name of the model: mediapipe, baseline3d, mano')
    parser.add_argument('-wp', '--model_weight_file_name', type=str, default='.pth', help='the file name of the weights of the trained model')
    parser.add_argument('-fs', '--fragment_size', type=int, default=1, help='the fragment size of the dataset, which is equal to the sequence length and the in_channel of the input')
    parser.add_argument('-sw', '--sliding_window', type=int, default=0, help='the sliding window size of the dataset')
    parser.add_argument('-ms', '--map_size', type=int, default=96, help='the size of the thermal array map (input) and the depth map')
    parser.add_argument('-hm', '--require_heatmap', type=int, default=0, help='require the 2d joints heatmap or not: 0,1')
    parser.add_argument('-fo', '--filter_out_no_hand', type=int, default=0, help='filter out the samples without hand in the dataset: 0,1')
    parser.add_argument('-mt', '--tapor_type', type=int, default=0, help='the type of tapor model: 0: small, 1: base1, 2: base2, 3: large1, 4: large2,5: base1_varaint1, 6: base1_variant2')
    parser.add_argument('-uf', '--up_sample_scale_factor', type=int, default=10, help='the up sampling scale factor of the spatial encoder of the tapor model')
    parser.add_argument('-v', '--video_flag', type=int, default=0, help='1: save video, 0: not save video')

    args = parser.parse_args()
    
    model_name = args.model_name
    model_weight_file_name = args.model_weight_file_name
    fragment_size = args.fragment_size
    map_size = args.map_size
    require_heatmap = args.require_heatmap
    filter_out_no_hand = args.filter_out_no_hand
    batch_size = 64
    sliding_window = args.sliding_window
    tapor_type = args.tapor_type
    up_sample_scale_factor = args.up_sample_scale_factor
    video_flag = args.video_flag

    data_path = 'Testset/'
    # file_names = os.listdir(data_path)
    file_names = ['Px_U1_R_hot_X_2.pkl', 'Px_U1_R_Warm_X_1.pkl', 'Px_U1_R_Warm_X_6.pkl', 
                  'Px_U1_R_Warm_X_7.pkl', 'Px_U1_R_normal_X_3.pkl', 'Px_U1_R_normal_X_4.pkl', 
                  'Px_U1_R_normal_X_5.pkl', 'Px_U1_R_normal_X_6.pkl', 'Px_U1_R_scold_X_1.pkl', 
                  'Px_U1_R_scold_X_4.pkl', 'Px_U1_R_vcold_X_4.pkl', 'P2_U1_R_X_X_Temp_17_0.pkl',
                  'P2_U1_R_X_X_Temp_17_1.pkl', 'P2_U1_R_X_X_Temp_17_2.pkl', 'P2_U1_R_X_X_Temp_17_3.pkl', 
                  'P2_U1_R_X_X_Temp_17_4.pkl', 'P2_U2_R_X_X_Temp_17_0.pkl', 'P2_U2_R_X_X_Temp_17_3.pkl', 
                  'P2_U2_R_X_X_Temp_24_1.pkl', 'P2_U2_R_X_X_Temp_24_2.pkl', 'P2_U2_R_X_X_Temp_17_1.pkl',
                  'P2_U2_R_X_X_Temp_17_2.pkl', 'P2_U2_R_X_X_Temp_17_4.pkl', 'P2_U1_R_X_X_Temp_20_2.pkl',
                  'P2_U1_R_X_X_Temp_20_3.pkl', 'P2_U2_R_X_X_Temp_20_0.pkl', 'P2_U2_R_X_X_Temp_20_1.pkl', 
                  'P2_U2_R_X_X_Temp_20_2.pkl', 'P2_U2_R_X_X_Temp_22_0.pkl', 'P2_U2_R_X_X_Temp_22_4.pkl', 
                  'P2_U1_R_X_X_Temp_24_0.pkl', 'P2_U1_R_X_X_Temp_24_4.pkl', 'P2_U2_R_X_X_Temp_24_0.pkl',
                  'P2_U2_R_X_X_Temp_24_3.pkl', 'P2_U2_R_X_X_Temp_24_4.pkl', 'P2_U1_R_X_X_Temp_26_4.pkl', 
                  'P2_U2_R_X_X_Temp_28_2.pkl', 'P2_U2_R_X_X_Temp_30_0.pkl', 'P2_U2_R_X_X_Temp_30_2.pkl', 
                  'P2_U2_R_X_X_Temp_30_3.pkl', 'P2_U2_R_X_X_Temp_32_0.pkl', 'P2_U2_R_X_X_Temp_32_1.pkl', 
                  'P2_U1_R_X_X_Temp_34_1.pkl', 'P2_U2_R_X_X_Temp_36_3.pkl', 'P1_empty_R_X_X_0.pkl', 
                  'P1_empty_R_X_X_1.pkl', 'P1_empty_R_X_X_2.pkl', 'P1_empty_R_X_X_3.pkl', 'P1_body_R_X_X_0.pkl',
                  'P1_body_R_X_X_2.pkl', 'P1_body_R_X_X_3.pkl', 'P1_laptop_R_X_X_0.pkl', 'P1_laptop_R_X_X_1.pkl',
                  'P1_laptop_R_X_X_3.pkl', 'P1_display_R_X_X_1.pkl', 'P1_display_R_X_X_2.pkl', 'P1_display_R_X_X_3.pkl', 
                  'P1_hotbottle_R_X_X_0.pkl', 'P1_hotbottle_R_X_X_2.pkl', 'P1_light_R_X_X_1.pkl', 'P1_light_R_X_X_2.pkl', 
                  'P1_light_R_X_X_3.pkl', 'Px_U2_R_LightCond6_X_2.pkl', 'Px_U2_R_LightCond6_X_4.pkl', 'Px_U1_R_LightConds7_X_1.pkl',
                  'Px_U1_R_LightConds7_X_7.pkl', 'P1_U10_L_X_X_0.pkl', 'P1_U10_L_X_X_1.pkl', 'P1_U10_R_X_X_1.pkl',
                  'P1_U1_L_X_X_1.pkl', 'P1_U1_L_X_X_2.pkl', 'P1_U1_R_Cover_coolwater_0.pkl', 'P1_U1_R_Cover_coolwater_1.pkl', 
                  'P1_U1_R_Cover_flour_0.pkl', 'P1_U1_R_Cover_flour_1.pkl', 'P1_U1_R_Cover_foam_0.pkl', 'P1_U1_R_Cover_foam_1.pkl',
                  'P1_U1_R_Cover_no_0.pkl', 'P1_U1_R_Cover_no_1.pkl', 'P1_U1_R_Cover_oil_0.pkl', 'P1_U1_R_Cover_oil_1.pkl',
                  'P1_U1_R_Cover_warmwater_0.pkl', 'P1_U1_R_Cover_warmwater_1.pkl', 'P1_U1_L_thick1_0.pkl', 'P1_U1_L_thick1_1.pkl',
                  'P1_U1_L_thick1_10.pkl', 'P1_U1_L_thick1_2.pkl', 'P1_U1_L_thick1_3.pkl', 'P1_U1_L_thick1_4.pkl', 'P1_U1_L_thick1_5.pkl', 
                  'P1_U1_L_thick1_6.pkl', 'P1_U1_L_thick1_7.pkl', 'P1_U1_L_thick1_8.pkl', 'P1_U1_L_thick1_9.pkl', 'P1_U1_R_thick0_1.pkl', 
                  'P1_U1_R_thick0_10.pkl', 'P1_U1_R_thick0_2.pkl', 'P1_U1_R_thick0_7.pkl', 'P1_U1_R_thick0_9.pkl', 'P1_U1_R_thick1_0.pkl', 
                  'P1_U1_R_thick1_1.pkl', 'P1_U1_R_thick1_10.pkl', 'P1_U1_R_thick1_2.pkl', 'P1_U1_R_thick1_3.pkl', 'P1_U1_R_thick1_5.pkl', 
                  'P1_U1_R_thick1_7.pkl', 'P1_U1_R_thick1_8.pkl', 'P1_U1_R_thick1_9.pkl', 'P1_U1_cover_medicalGlove_0.pkl', 
                  'P1_U1_cover_medicalGlove_1.pkl', 'P1_U1_cover_medicalGlove_10.pkl', 'P1_U1_cover_medicalGlove_2.pkl', 
                  'P1_U1_cover_medicalGlove_3.pkl', 'P1_U1_cover_medicalGlove_4.pkl', 'P1_U1_cover_medicalGlove_6.pkl',
                  'P1_U1_cover_medicalGlove_7.pkl', 'P1_U1_cover_medicalGlove_8.pkl', 'P1_U1_cover_medicalGlove_9.pkl', 
                  'P1_UX_cover_nail_0.pkl', 'P1_UX_cover_nail_2.pkl', 'P1_UX_cover_nail_3.pkl', 'P1_UX_cover_nail_4.pkl',
                  'P1_UX_cover_nail_6.pkl', 'P1_UX_cover_nail_7.pkl', 'P1_U1_R_Dist_Dist_0.pkl', 'P1_U1_R_Dist_Dist_1.pkl', 
                  'P1_U1_R_LightCond0_X_0.pkl', 'P1_U1_R_LightCond0_X_1.pkl', 'P1_U1_R_LightCond1_X_0.pkl', 'P1_U1_R_LightCond1_X_1.pkl',
                  'P1_U1_R_LightCond2_X_0.pkl', 'P1_U1_R_LightCond2_X_1.pkl', 'P1_U1_R_LightCond3_X_0.pkl', 'P1_U1_R_LightCond3_X_1.pkl',
                  'P1_U1_R_X_X_1.pkl', 'P1_U2_R_Cover_glove_1.pkl', 'P1_U1_R_Cover_ring_0.pkl', 'P1_U1_R_Cover_ring_1.pkl', 
                  'P1_U2_R_Cover_coolwater_0.pkl', 'P1_U2_R_Cover_coolwater_1.pkl', 'P1_U2_R_Cover_flour_0.pkl', 'P1_U2_R_Cover_flour_1.pkl', 
                  'P1_U2_R_Cover_foam_0.pkl', 'P1_U2_R_Cover_foam_1.pkl', 'P1_U2_R_Cover_no_0.pkl', 'P1_U2_R_Cover_no_1.pkl', 'P1_U2_R_Cover_oil_0.pkl',
                  'P1_U2_R_Cover_oil_1.pkl', 'P1_U2_R_Cover_warmwater_0.pkl', 'P1_U2_R_Cover_warmwater_1.pkl', 'P1_U2_R_Dist_Dist_0.pkl', 'P1_U2_R_Dist_Dist_1.pkl', 
                  'P1_U2_R_Dist_Dist_2.pkl', 'P1_U2_R_LightCond0_X_0.pkl', 'P1_U2_R_LightCond0_X_1.pkl', 'P1_U2_R_LightCond1_X_0.pkl', 'P1_U2_R_LightCond1_X_1.pkl',
                  'P1_U2_R_LightCond2_X_0.pkl', 'P1_U2_R_LightCond2_X_1.pkl', 'P1_U2_R_LightCond3_X_0.pkl', 'P1_U2_R_LightCond3_X_1.pkl', 
                  'P1_U2_R_X_X_0.pkl', 'P1_U2_R_X_X_1.pkl', 'P1_U2_R_X_X_2.pkl', 'P1_U3_L_X_X_1.pkl',
                  'P1_U3_R_X_X_0.pkl', 'P1_U4_L_X_X_0.pkl', 'P1_U4_R_X_X_0.pkl', 'P1_U5_R_X_X_0.pkl', 'P1_U5_R_X_X_1.pkl', 'P1_U6_L_X_X_0.pkl',
                  'P1_U6_L_X_X_1.pkl', 'P1_U6_R_X_X_0.pkl', 'P1_U6_R_X_X_1.pkl', 'P1_U7_L_X_X_1.pkl', 
                  'P1_U8_L_X_X_0.pkl', 'P1_U8_R_X_X_0.pkl', 'P1_U8_R_X_X_1.pkl', 'P1_U9_R_X_X_0.pkl', 'P2_U1_L_X_X_0.pkl', 'P2_U1_L_X_X_2.pkl', 
                  'P2_U1_R_X_X_0.pkl', 'P2_U2_L_X_X_0.pkl', 'P2_U2_L_X_X_1.pkl', 'P2_U2_L_X_X_2.pkl', 'P2_U2_R_X_X_1.pkl', 'P2_U2_R_X_X_2.pkl',
                  'P3_U1_L_X_X_1.pkl', 'P3_U1_L_X_X_2.pkl', 'P3_U1_R_X_X_0.pkl', 'P3_U1_R_X_X_1.pkl', 'P3_U1_R_X_X_2.pkl', 'P5_U1_L_X_X_0.pkl',
                  'P5_U1_L_X_X_1.pkl', 'P5_U1_L_X_X_2.pkl', 'P5_U1_R_X_X_0.pkl', 
                  'P5_U1_R_X_X_1.pkl', 'P5_U2_L_X_X_0.pkl', 'P5_U2_L_X_X_1.pkl', 'P5_U2_L_X_X_2.pkl', 'P5_U2_R_X_X_0.pkl', 'P5_U2_R_X_X_1.pkl', 
                  'P5_U2_R_X_X_2.pkl', 'P6_U1_R_X_X_2.pkl', 'P6_U1_L_X_X_1.pkl', 'P6_U1_L_X_X_0.pkl', 'P6_U1_R_X_X_1.pkl']
    log_folder = 'TestDataset_logs/'
    
    model_weight_path = "weights/" + model_weight_file_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists(model_weight_path):
        print("The model weight file does not exist!")
        exit()
    
    localtime = time.localtime(time.time())
    if model_name == 'tapor':
        log_file_name = model_name + "_type" + str(tapor_type) + "_weights_" + model_weight_file_name  + "_" + str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday) + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)
    else:
        log_file_name = model_name + "_weights_" + model_weight_file_name  + "_" + str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday) + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)
    print("The log file name is: ",log_file_name)
    
    model = None
    if model_name == 'baseline3d':
        if map_size < 200:
            print("The baseline3d model only support map_size >= 200")
            map_size = 256
        if fragment_size != 1:
            print("The baseline3d model only support fragment_size = 1")
            fragment_size = 1
        model = ColorHandPose3D(in_channels=fragment_size, crop_size=map_size).to(device)
    elif model_name == 'mediapipe':
        if map_size != 256:
            print("The mediapipe model support map_size = 256 or some larger ones")
            map_size = 256
        if fragment_size != 1:
            print("The mediapipe model only support fragment_size = 1")
            fragment_size = 1
        model = BlazeHandLandmark(resolution=map_size, in_channels=fragment_size).to(device)
    elif model_name == 'mano':
        if require_heatmap != 0:
            print("The mano model does not need the 2d heatmap for supervision")
            require_heatmap = 0
        if fragment_size != 1:
            print("The mano model only support fragment_size = 1")
            fragment_size = 1
        model = Mano_MobileNetV2(batch_size=batch_size).to(device)
    elif model_name == 'tapor':
        if require_heatmap != 0:
            print("The mano model does not need the 2d heatmap for supervision")
            require_heatmap = 0
                
        if tapor_type == 0:
            config = small_setting
        elif tapor_type == 1:
            config = base1_setting
        elif tapor_type == 2:
            config = base2_setting
        elif tapor_type == 3:
            config = large1_setting
        elif tapor_type == 4:
            config = large2_setting
        elif tapor_type == 5:
            config = base1_setting_varaint1
        elif tapor_type == 6:
            config = base1_setting_varaint2
        else:
            raise ValueError('The tapor type is not supported!')
                
        spatial_encoder_param = config['spatial_encoder_param']
        keypoints_encoder_param = config['keypoints_encoder_param']
        cross_keypoints_fusion_param = config['cross_keypoints_fusion_param']
        temporal_keypoints_fusion_param = config['temporal_keypoints_fusion_param']
        handpose_encoder_param = config['handpose_encoder_param']
        temporal_keypoints_fusion_param['num_history'] = fragment_size-1
        if up_sample_scale_factor != 10:
            spatial_encoder_param['up_sample_scale_factor'] = up_sample_scale_factor    

        model = Tapor(spatial_encoder_param, 
                 keypoints_encoder_param, 
                 cross_keypoints_fusion_param, 
                 temporal_keypoints_fusion_param,
                 handpose_encoder_param,
                 input_width=32, 
                 input_height=24,
                 batch_size = batch_size,
                 train=True,
                 device=device).to(device)
    else:
        raise ValueError('The model is not supported!')
    model.load_state_dict(torch.load(model_weight_path))
    
    # processing all the files one by one
    result_dict = {
        'infernece_file_name': [],
        'failed_file_names': [],
        'predicts': [],
        'labels': [],
    }
    all_predicts = []
    all_labels = []
    all_input_thermal_frame = []
    failed_file_names = []
    file_names_for_visualization = []
    for file_name in tqdm(file_names):
        try:
            file_name_now = [file_name]
            dataset = iHand_dataset(data_path, file_name_now,fragment_size,map_size = (map_size,map_size), require_heatmap=require_heatmap, filter_out_no_hand=filter_out_no_hand,sliding_window = sliding_window)
            print("The size of the dataset is: ",len(dataset))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last= True)
            predicts_dict,labels_dict = inference(model_name,model,dataloader,device)
            input_thermal_frame = np.concatenate(labels_dict['thermal_map'])
            predicts = np.concatenate(predicts_dict['joints_3d'])
            labels = np.concatenate(labels_dict['joints_3d'])          
            all_predicts.append(predicts)
            all_labels.append(labels)  
            all_input_thermal_frame.append(input_thermal_frame)    
            file_names_for_visualization += [file_name] * len(input_thermal_frame)    
            result_dict['infernece_file_name'].append(file_name)
            result_dict['predicts'].append(predicts)
            result_dict['labels'].append(labels)
            # calculating the mse and mae between the predicted landmarks and the ground truth landmarks
            print("The shape of the predicts and labels are: ")
            print(predicts.shape)
            print(labels.shape)
            average_mse_loss = np.mean((predicts - labels) ** 2)
            average_mae_loss = np.mean(np.abs(predicts - labels))
            per_dimention_mae_loss = np.mean(np.abs(predicts - labels),axis=0)
            per_dimention_mae_loss = np.mean(per_dimention_mae_loss,axis=0)
            per_dimention_mae_loss = per_dimention_mae_loss.tolist()
            mean_joints_error, mean_root_drift_error = calculate_shift_error(predicts, labels)
            mean_joints_error = mean_joints_error.tolist()
            mean_joints_error_str = [str(i) for i in mean_joints_error]
            mean_joints_error_str = ','.join(mean_joints_error_str)
            mpjpe = calculate_mpjpe(predicts, labels)
            pck_5mm = calculate_pck(predicts, labels, threshold=0.005)
            pck_1cm = calculate_pck(predicts, labels, threshold=0.01)
            pck_2cm = calculate_pck(predicts, labels, threshold=0.02)
            pck_3cm = calculate_pck(predicts, labels, threshold=0.03)
            pck_4cm = calculate_pck(predicts, labels, threshold=0.04)
            pck_5cm = calculate_pck(predicts, labels, threshold=0.05)
            pck_6cm = calculate_pck(predicts, labels, threshold=0.06)
            pck_7cm = calculate_pck(predicts, labels, threshold=0.07)
            pck_8cm = calculate_pck(predicts, labels, threshold=0.08)
            pck_9cm = calculate_pck(predicts, labels, threshold=0.09)
            print(f'Average MSE Loss(testset) {average_mse_loss:.10f}')
            print(f'Average MAE Loss(testset) {average_mae_loss:.10f}')
            print(f'Per Dimention MAE Loss(testset) {per_dimention_mae_loss}')
            print('Mean joints error:' + mean_joints_error_str + '\n')
            print(f'Mean root drift error: {mean_root_drift_error:.10f}')
            print(f'MPJPE {mpjpe:.10f}')
            print(f'PCK@5mm {pck_5mm:.10f}')
            print(f'PCK@1cm {pck_1cm:.10f}')
            print(f'PCK@2cm {pck_2cm:.10f}')
            print(f'PCK@3cm {pck_3cm:.10f}')
            print(f'PCK@4cm {pck_4cm:.10f}')
            print(f'PCK@5cm {pck_5cm:.10f}')
            print(f'PCK@6cm {pck_6cm:.10f}')
            print(f'PCK@7cm {pck_7cm:.10f}')
            print(f'PCK@8cm {pck_8cm:.10f}')
            print(f'PCK@9cm {pck_9cm:.10f}')
        except:
            failed_file_names.append(file_name)
            result_dict['failed_file_names'].append(file_name)
            print("The file: ",file_name," is not processed correctly!")
            continue
            
    # create a log file to write the above results
    with open(log_folder + log_file_name + '_log.txt', 'w') as f:
        f.write('Parameters: \n')
        f.write('The model name is: ' + model_name + '\n')
        f.write('The model type of tapor is: ' + str(tapor_type) + '\n')
        f.write('The model weight file name is: ' + model_weight_file_name + '\n')
        f.write('The fragment size is: ' + str(fragment_size) + '\n')
        f.write('The map size is: ' + str(map_size) + '\n')
        f.write('The require heatmap is: ' + str(require_heatmap) + '\n')
        f.write('The filter out no hand is: ' + str(filter_out_no_hand) + '\n')        
        f.write('The failed file names are: \n')
        for file_name in failed_file_names:
            f.write(file_name + '\n')
    # save the results
    print("The results are saved in: ",log_folder + log_file_name + '_log.txt')
    pickle.dump(result_dict,open(log_folder + log_file_name + '_result_dict.pkl','wb'))
    # create a video to visualize the results
    all_predicts = np.concatenate(all_predicts)
    all_labels = np.concatenate(all_labels)
    all_input_thermal_frame = np.concatenate(all_input_thermal_frame)
    if video_flag == 1:
        print("Generating the visualization video...")
        visualization_results_video(log_folder, log_file_name,all_input_thermal_frame, all_labels, all_predicts,(256*3,256*3), file_names_for_visualization)
        print("The visualization video is saved in: ",log_folder + log_file_name +'_'+"inference.mp4")
    else:
        print("The video is not generate!")
    print("The inference is done!")
