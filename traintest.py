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
from torch.utils.tensorboard import SummaryWriter
from dataset import iHand_dataset, RandomOrthogonalRotation
from models import BlazeHandLandmark, ColorHandPose3D, Mano_MobileNetV2, Tapor
from utils import draw_landmarks, draw_3D_landmarks, generate_heatmaps, calculate_mpjpe, calculate_pck, calculate_shift_error,BoneLengthLoss, KinematicChainLoss
from tapor_model_config import small_setting, base1_setting, base2_setting, large1_setting, large2_setting, base1_setting_varaint1, base1_setting_varaint2
import os
import timm
from timm.scheduler import CosineLRScheduler

tqdm_disable = False

def get_gpu_ids(i):
    gpu_ids = []
    gpu_info = os.popen("nvidia-smi -L").readlines()
    for line in gpu_info:
        # print(line)
        ids = line.split("UUID: ")[-1].strip(" ()\n")
        if ids.startswith("GPU"):
            continue
        gpu_ids.append(ids)
    if i >= len(gpu_ids):
        print("The number of the gpu is not enough! using the 0 by default")
        return gpu_ids[0]
    return gpu_ids[i]

tensorboard_folder = 'Runs/'

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
        for i, input in enumerate(tqdm(data_loader, disable= tqdm_disable)):
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
                labels_all['keypoints_scoremap'].append(heatmap_label.cpu().numpy())
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
                landmarks, current_kp_feat, sp_feat, kp_feat, attention_map = model(thermal_map) # (batch_size, 21, 3), (batch_size), (batch_size)
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
                labels_all['joints_3d'].append(l_3d_joint.cpu().numpy())
                labels_all['l_3d_flag'].append(l_3d_flag.cpu().numpy())
                labels_all['joints_2d'].append(l_2d_joint.cpu().numpy())
                labels_all['l_2d_flag'].append(l_2d_flag.cpu().numpy())
                labels_all['l_hand_depth'].append(l_hand_depth.numpy())
                labels_all['l_left_right_flag'].append(l_left_right_flag.numpy())
                labels_all['ambient_temperature'].append(ambient_temperature.numpy())  
                 
    return predicts_all,labels_all


def train(log_file_name,model_name,model,epochs,train_loader,validation_loader,filter_out_no_hand,device,loss_function,optimizer,scheduler=None,max_norm = 10):
    best_loss = float('inf')
    best_model_wts = None
    best_model_epoch = 0
    
    if 'j' in loss_function:
        if filter_out_no_hand:
            criterion_joints = nn.MSELoss()
        else:
            criterion_joints = nn.MSELoss(reduction='none') # to handle the case that there is no hand in the map
    else:
        criterion_joints = None
        
    if 'b' in loss_function:
        criterion_bone = BoneLengthLoss()
    else:
        criterion_bone = None
        
    if 'k' in loss_function:
        criterion_kinematic = KinematicChainLoss(device)
    else:
        criterion_kinematic = None
    
    if model_name == 'baseline3d':    # this baseline need the heatmap loss
        criterion_heatmap = nn.MSELoss(reduction='mean')

    writer = SummaryWriter(tensorboard_folder+ log_file_name + "_" + model_name)
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        loss_epoch = 0
        for i, input in enumerate(tqdm(train_loader ,disable= tqdm_disable)):
            thermal_map, ambient_temperature, l_depth_map, l_2d_joint, l_2d_flag, l_3d_joint, l_3d_flag, l_hand_depth, l_left_right_flag, heatmap_label = input
            l_2d_flag = l_2d_flag.float().to(device)  # (batch_size, 1)
            l_2d_joint = l_2d_joint.squeeze().float().to(device) # (batch_size,c ,21, 3), the last dim is unuseful; if c=1, then  (batch_size ,21, 3)
            l_3d_flag = l_3d_flag.float().to(device)  # (batch_size, 1)
            l_3d_joint = l_3d_joint.squeeze().float().to(device) # (batch_size, c, 21, 3); if c=1, then  (batch_size ,21, 3)
            thermal_map = thermal_map.float().to(device)  # (batch_size, c, 256, 256)
            loss = 0
            # with torch.autograd.detect_anomaly():
            if model_name == 'baseline3d':
                l_left_right_flag = l_left_right_flag.squeeze().float().to(device)  # (batch_size, 2)
                coord_can, keypoints_scoremap = model(thermal_map, l_left_right_flag) # (batch_size, 21, 3), (batch_size, 21, 256, 256)
                if criterion_bone is not None:
                    loss_b = criterion_bone(coord_can, l_3d_joint)
                    loss += loss_b
                if criterion_kinematic is not None:
                    loss_k = criterion_kinematic(coord_can, l_3d_joint)
                    loss += loss_k
                if criterion_joints is not None:
                    if filter_out_no_hand:    # all the samples have hands
                        loss_j = criterion_joints(coord_can, l_3d_joint) # (batch_size, 21, 3)
                    else:
                        l_3d_flag = l_3d_flag.unsqueeze(2).expand(l_3d_joint.shape) # (batch_size, 21, 3)
                        loss_j = criterion_joints(coord_can, l_3d_joint) # (batch_size, 21, 3)
                        loss_j = torch.mean(loss_j * l_3d_flag)
                    loss += loss_j
                heatmap_label = heatmap_label.float().to(device)  # (batch_size, 21, h,w)
                loss_h = criterion_heatmap(keypoints_scoremap.squeeze(), heatmap_label.squeeze())
                loss += loss_h
            elif model_name == 'mano':
                l_left_right_flag = l_left_right_flag.squeeze().float().to(device)  # (batch_size, 2)
                joints_left, joints_right = model(thermal_map) # (batch_size, 21, 3), (batch_size, 21, 256, 256)
                # loss1: the loss of the joints, estimated 3d hand pose
                if criterion_bone is not None:
                    loss_b = criterion_bone(joints_left, l_3d_joint)
                    loss += loss_b
                if criterion_kinematic is not None:
                    loss_k = criterion_kinematic(joints_left, l_3d_joint)
                    loss += loss_k
                if filter_out_no_hand:
                    loss1 = criterion_joints(joints_left, l_3d_joint) 
                    loss2 = criterion_joints(joints_right, l_3d_joint) # (batch_size, 21, 3)
                    loss_j = l_left_right_flag[:, 0].unsqueeze(1).unsqueeze(2) * loss1 + \
                                    l_left_right_flag[:, 1].unsqueeze(1).unsqueeze(2) * loss2
                    loss_j = loss_j.mean()
                    loss += loss_j
                else:
                    loss1 = criterion_joints(joints_left, l_3d_joint) 
                    loss2 = criterion_joints(joints_right, l_3d_joint) # (batch_size, 21, 3)
                    loss_j = l_left_right_flag[:, 0].unsqueeze(1).unsqueeze(2) * loss1 + \
                                    l_left_right_flag[:, 1].unsqueeze(1).unsqueeze(2) * loss2
                    loss_j = torch.mean(loss_j * l_3d_flag)
                    loss += loss_j
            elif model_name == 'mediapipe':
                landmarks, hand_flag, handed = model(thermal_map) # (batch_size, 21, 3), (batch_size), (batch_size)
                if criterion_bone is not None:
                    loss_b = criterion_bone(landmarks, l_3d_joint)
                    loss += loss_b
                if criterion_kinematic is not None:
                    loss_k = criterion_kinematic(landmarks, l_3d_joint)
                    loss += loss_k
                if criterion_joints is not None:
                    if filter_out_no_hand:
                        loss_j = criterion_joints(landmarks, l_3d_joint)
                        loss += loss_j
                    else:
                        l_3d_flag_temp = l_3d_flag.clone()
                        l_3d_flag_temp = l_3d_flag_temp.unsqueeze(2).expand(l_3d_joint.shape) # (batch_size, 21, 3)
                        loss_j = criterion_joints(landmarks, l_3d_joint) # (batch_size, 21, 3)
                        loss_j = torch.mean(loss_j * l_3d_flag_temp) 
                        loss += loss_j
            elif model_name == 'tapor':
                landmarks, current_kp_feat, sp_feat, kp_feat,attention_map = model(thermal_map) # (batch_size, 21, 3), (batch_size), (batch_size)
                # l_3d_joint shape is four dim: (batch_size, c, 21, 3), we only use the last element of c
                if l_3d_joint.shape[1] == 21:
                    pass
                else:
                    l_3d_joint = l_3d_joint[:, -1, :, :]  # (batch_size, 21, 3)
                    
                if criterion_bone is not None:
                    loss_b = criterion_bone(landmarks, l_3d_joint)
                    loss += loss_b
                if criterion_kinematic is not None:
                    loss_k = criterion_kinematic(landmarks, l_3d_joint)
                    loss += loss_k
                if criterion_joints is not None:
                    if filter_out_no_hand:
                        loss_j = criterion_joints(landmarks, l_3d_joint)
                    else:
                        l_3d_flag_temp = l_3d_flag.clone()
                        l_3d_flag_temp = l_3d_flag_temp.unsqueeze(2).expand(l_3d_joint.shape) 
                        loss_j = criterion_joints(landmarks, l_3d_joint) 
                        loss_j = torch.mean(loss_j * l_3d_flag_temp) 
                    loss += loss_j
            loss_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm, norm_type=2)
            optimizer.step()
            writer.add_scalar('train loss (per batch)', loss.item(), epoch * len(train_loader) + i)
            if'j' in loss_function:
                writer.add_scalar('train joint loss (per batch)', loss_j.item(), epoch * len(train_loader) + i)
            if'b' in loss_function:
                writer.add_scalar('train bone loss (per batch)', loss_b.item(), epoch * len(train_loader) + i)
            if'k' in loss_function:
                writer.add_scalar('train kinematic loss (per batch)', loss_k.item(), epoch * len(train_loader) + i)
        if scheduler is not None:
            scheduler.step(epoch)
        print(f'Epoch {epoch}, Average Loss(train set) {loss_epoch/ len(train_loader):.10f}')
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train loss (average)', loss_epoch/ len(train_loader), epoch)
        predicts,labels = inference(model_name,model,validation_loader,device)
        average_mse_loss = np.mean((np.concatenate(predicts['joints_3d']) - np.concatenate(labels['joints_3d'])) ** 2)
        average_mae_loss = np.mean(np.abs(np.concatenate(predicts['joints_3d']) - np.concatenate(labels['joints_3d'])))
        writer.add_scalar('validation mse loss (average)', average_mse_loss, epoch)
        writer.add_scalar('validation mae loss (average)',average_mae_loss,epoch)
        print(f'Epoch {epoch}, Average MSE Loss(val set) {average_mse_loss:.10f}')
        print(f'Epoch {epoch}, Average MAE Loss(val set) {average_mae_loss:.10f}')
        
        if average_mae_loss<best_loss:
            best_model_wts = model.state_dict()
            best_loss = average_mae_loss
            best_model_epoch = epoch
        else:
            if epoch - best_model_epoch >= 40:    
                print('early stop with best model at epoch: ',best_model_epoch)
                break
        # if the loss is nan, then stop the training
        if np.isnan(average_mae_loss):
            print('early nan stop with best model at epoch: ',best_model_epoch)
            break
    writer.close()
    saved_model_path = 'weights/'+ log_file_name + "_" + model_name + '.pth'
    torch.save(best_model_wts, saved_model_path)
    return saved_model_path, best_model_epoch



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training and Testing')
    parser.add_argument('-m', '--model_name', type=str, default='baseline3d', help='the name of the model: mediapipe, baseline3d, mano, tapor')
    parser.add_argument('-wp', '--model_weight_file_name', type=str, default='no_weight', help='the file name of the weights of the trained model for resuming training')
    parser.add_argument('-mt', '--tapor_type', type=int, default=0, help='the type of tapor model: 0: small, 1: base1, 2: base2, 3: large1, 4: large2,5: base1_varaint1, 6: base1_variant2')
    parser.add_argument('-uf', '--up_sample_scale_factor', type=int, default=10, help='the up sampling scale factor of the spatial encoder of the tapor model')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='the batch size')
    parser.add_argument('-fs', '--fragment_size', type=int, default=10, help='the fragment size of the dataset, which is equal to the sequence length and the in_channel of the input')
    parser.add_argument('-ms', '--map_size', type=int, default=96, help='the size of the thermal array map (input) and the depth map ; 0 means keeps the original size')
    parser.add_argument('-hm', '--require_heatmap', type=int, default=0, help='require the 2d joints heatmap or not: 0,1')
    parser.add_argument('-fo', '--filter_out_no_hand', type=int, default=0, help='filter out the samples without hand in the dataset: 0,1')
    parser.add_argument('-t', '--transform', type=int, default=0, help='whether to transform the data')
    parser.add_argument('-ls', '--loss', type=str, default='j', help='the loss function: j: joint loss, b: bone length loss, k: kinematic chain loss')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='the learning rate')
    parser.add_argument('-s', '--scheduler_step_ratio', type=int, default=0, help='0: no step scheduler; others: the stride coefficeient of the step scheduler')
    parser.add_argument('-c', '--cuda_index', type=int, default=0, help='the index of the cuda device')
    parser.add_argument('-tqdm', '--tqdm_d', type=int, default=0, help='whether disable the tqdm, 0: disable, 1: enable')
    parser.add_argument('-ntr', '--new_train_recipe', type=int, default=0, help='whether to use the new training recipe: adamW with cosine lr scheduler, 0: no, 1: yes')
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)
    
    model_name = args.model_name
    model_weight_file_name = args.model_weight_file_name
    epochs = args.epochs   
    batch_size = args.batch_size
    fragment_size = args.fragment_size
    map_size = args.map_size
    transform = args.transform
    learning_rate = args.learning_rate
    require_heatmap = args.require_heatmap
    filter_out_no_hand = args.filter_out_no_hand
    scheduler_step_ratio = args.scheduler_step_ratio
    tapor_type = args.tapor_type
    up_sample_scale_factor = args.up_sample_scale_factor
    cuda_index = args.cuda_index
    loss_function = args.loss
    new_train_recipe = args.new_train_recipe
    if args.tqdm_d == 0:
        tqdm_disable = True
    else:
        tqdm_disable = False
    data_path = 'Trainset/'
    
    file_names = [
	 'P1_U1_L_1_D_0.pkl',
	 'P1_U1_L_1_Z_0.pkl',
	 'P1_U1_L_2_O_0.pkl',
	 'P1_U1_L_2_S_0.pkl',
	 'P1_U1_L_3_T_0.pkl',
	 'P1_U1_L_4_O_0.pkl',
	 'P1_U1_L_4_Z_0.pkl',
	 'P1_U1_L_5_D_0.pkl',
	 'P1_U1_L_5_T_0.pkl',
	 'P1_U1_L_6_O_0.pkl',
	 'P1_U1_L_7_S_0.pkl',
	 'P1_U1_L_7_T_0.pkl',
	 'P1_U1_L_8_D_0.pkl',
	 'P1_U1_L_8_Z_0.pkl',
	 'P1_U1_L_9_O_0.pkl',
	 'P1_U1_L_9_S_0.pkl',
	 'P1_U1_L_X_X_0.pkl',
	 'P1_U1_L_X_X_1.pkl',
	 'P1_U1_L_X_X_2.pkl',
	 'P1_U1_L_X_X_3.pkl',
	 'P1_U1_L_X_X_4.pkl',
	 'P1_U1_R_0_D_0.pkl',
	 'P1_U1_R_0_O_0.pkl',
	 'P1_U1_R_0_Z_0.pkl',
	 'P1_U1_R_1_D_1.pkl',
	 'P1_U1_R_1_S_1.pkl',
	 'P1_U1_R_1_T_1.pkl',
	 'P1_U1_R_1_Z_1.pkl',
	 'P1_U1_R_2_S_0.pkl',
	 'P1_U1_R_2_S_1.pkl',
	 'P1_U1_R_3_D_0.pkl',
	 'P1_U1_R_3_T_0.pkl',
	 'P1_U1_R_4_O_0.pkl',
	 'P1_U1_R_4_Z_0.pkl',
	 'P1_U1_R_5_S_0.pkl',
	 'P1_U1_R_5_T_0.pkl',
	 'P1_U1_R_6_D_0.pkl',
	 'P1_U1_R_7_O_0.pkl',
	 'P1_U1_R_7_S_0.pkl',
	 'P1_U1_R_8_D_0.pkl',
	 'P1_U1_R_8_T_0.pkl',
	 'P1_U1_R_9_O_0.pkl',
	 'P1_U1_R_9_Z_0.pkl',
	 'P1_U1_R_X_X_0.pkl',
	 'P1_U1_R_X_X_1.pkl',
	 'P1_U1_R_X_X_2.pkl',
	 'P1_U1_R_X_X_3.pkl',
	 'P1_U1_R_X_X_4.pkl',
	 'P1_U1_R_X_X_5.pkl',
	 'P1_U2_L_0_O_0.pkl',
	 'P1_U2_L_0_T_0.pkl',
	 'P1_U2_L_0_Z_0.pkl',
	 'P1_U2_L_0_Z_1.pkl',
	 'P1_U2_L_1_X_0.pkl',
	 'P1_U2_L_2_O_0.pkl',
	 'P1_U2_L_3_T_0.pkl',
	 'P1_U2_L_6_D_0.pkl',
	 'P1_U2_L_9_T_0.pkl',
	 'P1_U2_L_X_X_0.pkl',
	 'P1_U2_L_X_X_1.pkl',
	 'P1_U2_L_X_X_4.pkl',
	 'P1_U2_R_0_D_0.pkl',
	 'P1_U2_R_0_D_1.pkl',
	 'P1_U2_R_0_O_0.pkl',
	 'P1_U2_R_0_T_0.pkl',
	 'P1_U2_R_0_T_2.pkl',
	 'P1_U2_R_0_X_1.pkl',
	 'P1_U2_R_0_Z_0.pkl',
	 'P1_U2_R_1_D_1.pkl',
	 'P1_U2_R_1_O_0.pkl',
	 'P1_U2_R_1_T_0.pkl',
	 'P1_U2_R_1_T_1.pkl',
	 'P1_U2_R_1_X_0.pkl',
	 'P1_U2_R_1_Z_0.pkl',
	 'P1_U2_R_1_Z_1.pkl',
	 'P1_U2_R_2_O_0.pkl',
	 'P1_U2_R_2_X_0.pkl',
	 'P1_U2_R_2_X_1.pkl',
	 'P1_U2_R_3_T_0.pkl',
	 'P1_U2_R_3_X_0.pkl',
	 'P1_U2_R_3_X_1.pkl',
	 'P1_U2_R_4_X_1.pkl',
	 'P1_U2_R_4_Z_0.pkl',
	 'P1_U2_R_4_Z_1.pkl',
	 'P1_U2_R_5_S_0.pkl',
	 'P1_U2_R_5_S_1.pkl',
	 'P1_U2_R_5_X_0.pkl',
	 'P1_U2_R_5_X_1.pkl',
	 'P1_U2_R_6_D_0.pkl',
	 'P1_U2_R_6_X_0.pkl',
	 'P1_U2_R_6_X_1.pkl',
	 'P1_U2_R_7_O_0.pkl',
	 'P1_U2_R_7_X_1.pkl',
	 'P1_U2_R_7_Z_0.pkl',
	 'P1_U2_R_8_X_1.pkl',
	 'P1_U2_R_8_Z_0.pkl',
	 'P1_U2_R_8_Z_1.pkl',
	 'P1_U2_R_9_T_0.pkl',
	 'P1_U2_R_9_X_0.pkl',
	 'P1_U2_R_X_X_3.pkl',
	 'P1_U2_R_X_X_4.pkl',
	 'P1_U2_R_X_X_5.pkl',
	 'P1_U2_R_X_X_6.pkl',
	 'P1_U3_L_X_X_0.pkl',
	 'P1_U3_L_X_X_1.pkl',
	 'P1_U3_L_X_X_2.pkl',
	 'P1_U3_R_X_X_1.pkl',
	 'P1_U3_R_X_X_2.pkl',
	 'P1_U4_L_X_X_0.pkl',
	 'P1_U4_R_X_X_0.pkl',
	 'P1_U4_R_X_X_1.pkl',
	 'P1_U4_R_X_X_2.pkl',
	 'P1_U5_L_X_X_0.pkl',
	 'P1_U5_L_X_X_1.pkl',
	 'P1_U5_R_X_X_0.pkl',
	 'P1_U5_R_X_X_1.pkl',
	 'P1_U5_R_X_X_2.pkl',
	 'P1_U6_L_X_X_0.pkl',
	 'P1_U6_L_X_X_2.pkl',
	 'P1_U6_R_X_X_0.pkl',
	 'P1_U6_R_X_X_1.pkl',
	 'P1_U6_R_X_X_2.pkl',
	 'P1_U6_R_X_X_3.pkl',
	 ]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_ids(cuda_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The device is: ",device)
    
    # fix the seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # saving the log of the training and testing process
    log_folder = 'LogTrainTest/'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if not os.path.exists('weights'):
        os.makedirs('weights')
    if not os.path.exists(tensorboard_folder):
        os.makedirs(tensorboard_folder)
    
    localtime = time.localtime(time.time())
    index_of_experiment = len(os.listdir(tensorboard_folder))
    # the name for both the log file and the tensorboard log
    log_file_name = str(index_of_experiment) + "_" + str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday) + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)
    print("The log file name is: ",log_file_name)
    model = None
    if model_name == 'baseline3d':
        if require_heatmap != 1:
            print("The baseline3d model need the 2d heatmap for supervision")
            require_heatmap = 1
        if map_size < 200:
            print("The baseline3d model only support map_size >= 200")
            map_size = 256
        if fragment_size != 1:
            print("The baseline3d model only support fragment_size = 1")
            fragment_size = 1
        model = ColorHandPose3D(in_channels=fragment_size, crop_size=map_size).to(device)
    elif model_name == 'mediapipe':
        if require_heatmap != 0:
            print("The mediapipe model does not need the 2d heatmap for supervision")
            require_heatmap = 0
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

    if new_train_recipe == 1:
        # using optimizer admW with cosine lr scheduler, the warm up is 2 epochs
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        warmup_epochs = 2
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epochs//4,
            lr_min=1e-6,  # Minimum LR
            warmup_lr_init=1e-6,  # Initial LR during warm-up
            warmup_t=warmup_epochs,  # Number of warm-up epochs
            cycle_limit=20,  # Number of LR cycles, 1 for a single decrease
            t_in_epochs=True,  # Interpret t_initial and warmup_t as epochs
        )
        print("Using the new training recipe: adamW with cosine lr scheduler")
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if scheduler_step_ratio == 0:
            scheduler = None
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//scheduler_step_ratio, gamma=0.05)
    
    # dataset initialization
    if transform:
        # if we want to predict the 3D hand pose, we can not use this data augmentation
        print("Warning: the data augmentation is not suitable for the 3D hand pose prediction!" )
        transform = RandomOrthogonalRotation()
    else:
        transform = None
    dataset = iHand_dataset(data_path, file_names,fragment_size,map_size = (map_size,map_size), transform=transform, require_heatmap=require_heatmap, filter_out_no_hand=filter_out_no_hand, shuffle_fragment=True)
    print("The size of the dataset is: ",len(dataset))
    train_size = int(0.8 * len(dataset))
    validation_size = int(0.1 * len(dataset))
    # just for initial test, the final test is carried out after the training in the inference file
    test_size = len(dataset) - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    if model_weight_file_name != 'no_weight':
        model_weight_path = 'weights/'+ model_weight_file_name
        if os.path.exists(model_weight_path):
            try:
                model.load_state_dict(torch.load(model_weight_path))
                print("resuming training with the weights: ",model_weight_file_name)
            except:
                # Check if CUDA is available and decide the device to use
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.load_state_dict(torch.load(model_weight_path, map_location=device))
                print("resuming training with the weights: ",model_weight_file_name)
        else:
            print("The weights file does not exist!")
            model_weight_file_name = 'no_weight'
    
    print("Start training!")
    # start training and validation
    saved_model_path, best_model_epoch = train(log_file_name,model_name,model,epochs,train_loader,validation_loader,filter_out_no_hand,device,loss_function,optimizer,scheduler)
    # testing the trained model
    try:
        model.load_state_dict(torch.load(saved_model_path))
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(saved_model_path, map_location=device))
    predicts_dict,labels_dict = inference(model_name,model,test_loader,device)
    input_thermal_frame = np.concatenate(labels_dict['thermal_map'])
    predicts = np.concatenate(predicts_dict['joints_3d'])
    labels = np.concatenate(labels_dict['joints_3d'])                    
    
    print("The shape of the predicts and labels are: ")
    print(predicts.shape)
    print(labels.shape)
    # calculating the mse and mae between the predicted landmarks and the ground truth landmarks
    average_mse_loss = np.mean((predicts - labels) ** 2)
    average_mae_loss = np.mean(np.abs(predicts - labels))
    per_dimention_mae_loss = np.mean(np.abs(predicts - labels),axis=0)
    per_dimention_mae_loss = np.mean(per_dimention_mae_loss,axis=0)
    per_dimention_mae_loss = per_dimention_mae_loss.tolist()
    print(f'Average MSE Loss(testset) {average_mse_loss:.10f}')
    print(f'Average MAE Loss(testset) {average_mae_loss:.10f}')
    print(f'Per Dimention MAE Loss(testset) {per_dimention_mae_loss}')
    
    mean_joints_error, mean_root_drift_error = calculate_shift_error(predicts, labels)
    mean_joints_error = mean_joints_error.tolist()
    mean_joints_error = [str(i) for i in mean_joints_error]
    mean_joints_error = ','.join(mean_joints_error)
    
    print('Mean joints error:'  +mean_joints_error + '\n')
    print(f'Mean root drift error: {mean_root_drift_error:.10f}')
    
    # calculating the mpjpe and pck between the predicted landmarks and the ground truth landmarks
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
    
    # create a log file to write the above results
    with open('LogTrainTest/' + log_file_name + '_log.txt', 'w') as f:
        f.write('The model name is: ' + model_name + '\n')
        f.write('The model type of tapor is: ' + str(tapor_type) + '\n')
        f.write('The model weight file name is: ' + model_weight_file_name + '\n')
        f.write('The best model is at epoch: ' + str(best_model_epoch) + '\n')
        f.write('The saved model path is: ' + saved_model_path + '\n')
        f.write('Average MSE Loss(testset): ' + str(average_mse_loss) + '\n')
        f.write('Average MAE Loss(testset): ' + str(average_mae_loss) + '\n')
        f.write('Per Dimention MAE Loss(testset): ' + str(per_dimention_mae_loss) + '\n')
        f.write('MPJPE: ' + str(mpjpe) + '\n')
        f.write('PCK@5mm: ' + str(pck_5mm) + '\n')
        f.write('PCK@1cm: ' + str(pck_1cm) + '\n')
        f.write('PCK@2cm: ' + str(pck_2cm) + '\n')
        f.write('PCK@3cm: ' + str(pck_3cm) + '\n')
        f.write('PCK@4cm: ' + str(pck_4cm) + '\n')
        f.write('PCK@5cm: ' + str(pck_5cm) + '\n')
        f.write('PCK@6cm: ' + str(pck_6cm) + '\n')
        f.write('PCK@7cm: ' + str(pck_7cm) + '\n')
        f.write('PCK@8cm: ' + str(pck_8cm) + '\n')
        f.write('PCK@9cm: ' + str(pck_9cm) + '\n')
        f.write('Mean joints error:' + mean_joints_error + '\n')
        f.write('Mean root drift error: ' + str(mean_root_drift_error) + '\n')
    # save the predicts and labels as a dict in pickle
    predicts_labels = {'input_thermal_map':  input_thermal_frame ,'predicts':predicts,'labels':labels}
    pickle.dump(predicts_labels,open('LogTrainTest/' + log_file_name + '_predicts_labels.pkl','wb'))
    
