# functions
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import math
import os
import torch.nn as nn

####################################### functions for visualization #######################################
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

def draw_landmarks(img, points, color=(0, 255, 0), size=1):
    connections =  HAND_CONNECTIONS
    points = np.array(points)[:,:2]
    points[:,0]*=img.shape[1]
    points[:,1]*=img.shape[0]
    
    for point in points:
        x, y = point
        if x< float('inf') and y<float('inf'):
            x, y = int(x), int(y)
        if x<0 or y<0 or x>img.shape[1] or y>img.shape[1]:
            continue
        cv2.circle(img, (x, y), size, color, thickness=size)
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        if x0<0 or y0<0 or x0>img.shape[1] or y0>img.shape[1] or x0<0 or y1<0 or x1>img.shape[1] or y1>img.shape[1]:
            continue
        cv2.line(img, (x0, y0), (x1, y1), (0,0,0), size)

def draw_3D_landmarks(points):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    height_default, dpi_default= (4.8, 300)
    lines = [[0,1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20], [0,5,9,13,17,0]]
    
    x,y,z = [],[],[]
    for id in lines[-1]:
        x.append(-points[id][0])
        y.append(-points[id][1])
        z.append(points[id][2])
    ax.plot(x,z,y,color='k', linestyle="-", linewidth=1)
    ax.scatter(x, z, y, s=10, c='r', marker='o')
    
    for line in lines[:5]:
        x,y,z = [],[],[]
        for id in line:
            x.append(-points[id][0])
            y.append(-points[id][1])
            z.append(points[id][2])
        ax.plot(x,z,y,color='g', linestyle="-", linewidth=1)
        ax.scatter(x, z, y, s=10, c='r', marker='o')
    
    # set the limits of the plot to the limits of the data
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(0, 0.6)
    ax.set_zlim(-0.2, 0.2)
    ax.set_xlabel('x Label')
    ax.set_ylabel('Depth Label')
    ax.set_zlabel('y Label')
    fig.canvas.draw()
    fig_str = fig.canvas.tostring_rgb()
    plot_img = np.frombuffer(fig_str, dtype=np.uint8).reshape((int(height_default*dpi_default), -1, 3)) #/ 255.0
    plt.close(fig)
    return plot_img


####################################### functions for measuring the performance #######################################
def calculate_shift_error(estimated_poses, ground_truth_poses):
    # estimated_poses: (batch_size, num_points, 3)
    # ground_truth_poses: (batch_size, num_points, 3)
    # first calculate the root joints distance
    # then, move the estimated_poses to the same position as the ground_truth_poses with the root joints distance
    # finally, calculate the mean relative error
    gt_root_joints = ground_truth_poses[:, 0, :] # shape (batch_size, 3)
    estimated_root_joints = estimated_poses[:, 0, :] # shape (batch_size, 3)
    root_drift = gt_root_joints - estimated_root_joints # shape (batch_size, 3)
    root_drift_n = root_drift[:, np.newaxis, :] # shape (batch_size, 1, 3)
    # expand root_drift to the same shape as estimated_poses
    # root_drift = root_drift_n.repeat(estimated_poses.shape[1], axis=1) # shape (batch_size, num_points, 3)
    shifted_estimated_poses = estimated_poses + root_drift_n # shape (batch_size, num_points, 3)
    
    joints_error = np.linalg.norm(shifted_estimated_poses - ground_truth_poses, axis=-1) # shape (batch_size, num_points)
    mean_joints_error = np.mean(joints_error, axis=0) # shape (batch_size,)
    
    root_drift_error = np.linalg.norm(root_drift, axis=-1) # shape (batch_size,)
    mean_root_drift_error = np.mean(root_drift_error)
    return mean_joints_error, mean_root_drift_error


def calculate_mpjpe(estimated_poses, ground_truth_poses):
    # estimated_poses: (batch_size, num_points, 3)
    # ground_truth_poses: (batch_size, num_points, 3)
    # reshape them to (batch_size*num_points, 3)
    estimated_poses = estimated_poses.reshape(-1, 3)
    ground_truth_poses = ground_truth_poses.reshape(-1, 3)
    errors = np.linalg.norm(estimated_poses - ground_truth_poses, axis=-1)
    mpjpe = np.mean(errors)
    return mpjpe
    
def calculate_pck(estimated_poses, ground_truth_poses, threshold, adjust_root_shift=False):
    if adjust_root_shift:
        gt_root_joints = ground_truth_poses[:, 0, :]
        estimated_root_joints = estimated_poses[:, 0, :]
        root_drift = gt_root_joints - estimated_root_joints
        root_drift_n = root_drift[:, np.newaxis, :]
        estimated_poses = estimated_poses + root_drift_n
    estimated_poses = estimated_poses.reshape(-1, 3)
    ground_truth_poses = ground_truth_poses.reshape(-1, 3)
    errors = np.linalg.norm(estimated_poses - ground_truth_poses, axis=-1)
    # calculate the number of correct keypoints that the error is less than the threshold
    num_correct = np.sum(errors < threshold)
    # calculate the number of keypoints
    num_total = len(errors)
    # calculate the pck
    pck = num_correct / num_total
    return pck
    
def calculate_mse_joints(estimated_poses, ground_truth_poses):
    estimated_poses = estimated_poses.reshape(-1, 3)
    ground_truth_poses = ground_truth_poses.reshape(-1, 3)
    errors = np.linalg.norm(estimated_poses - ground_truth_poses, axis=-1)
    mse = np.mean(errors**2, axis=-1)
    return mse

def calculate_mae_joints(estimated_poses, ground_truth_poses):
    estimated_poses = estimated_poses.reshape(-1, 3)
    ground_truth_poses = ground_truth_poses.reshape(-1, 3)
    errors = np.abs(estimated_poses - ground_truth_poses)
    mae = np.mean(errors, axis=-1)
    return mae

def calculate_mae_along_aixs(estimated_poses, ground_truth_poses, axis=0):
    estimated_poses_ = estimated_poses[:, :, axis]
    ground_truth_poses_ = ground_truth_poses[:, :, axis]
    errors = np.abs(estimated_poses_ - ground_truth_poses_)
    mae_along_aixs = np.mean(errors)
    return mae_along_aixs


##################################### functions for data representation #####################################
def generate_heatmaps(joints_2d, heatmap_size=(256, 256), sigma=10):
    """
    Generate heatmaps for a batch of coordinate points.
    :param joints_2d: Tensor of shape (batch_size, num_points, 2) containing normalized coordinates.
    :param heatmap_size: Size of the heatmap (width, height).
    :param sigma: Standard deviation for the Gaussian distribution.
    :return: Array of heatmaps, one for each batch.
    """
    batch_size, num_points, _ = joints_2d.shape
    heatmaps = np.zeros((batch_size, num_points, heatmap_size[0], heatmap_size[1]))
    x_axis = np.linspace(0, heatmap_size[0], heatmap_size[0])
    y_axis = np.linspace(0, heatmap_size[1], heatmap_size[1])
    X, Y = np.meshgrid(x_axis, y_axis)
    for i in range(batch_size):
        tmp = []
        for j in range(num_points):
            x, y = joints_2d[i, j]
            if x <0 or x>1:
                x = 0
            if y <0 or y>1:
                y = 0
            x_scaled = int(x * heatmap_size[0])
            y_scaled = int(y * heatmap_size[1])
            gaussian = np.exp(-((X - x_scaled)**2 + (Y - y_scaled)**2) / (2 * sigma**2))
            heatmaps[i, j] = gaussian
            heatmaps[i, j] = heatmaps[i, j] / np.max(heatmaps[i, j])        
    return heatmaps


# def generate_heatmaps(joints_2d, heatmap_size=(256, 256), sigma=10):
#     """
#     Generate heatmaps for a batch of coordinate points using vectorized operations for speedup.
#     :param joints_2d: Tensor of shape (batch_size, num_points, 2) containing normalized coordinates.
#     :param heatmap_size: Size of the heatmap (width, height).
#     :param sigma: Standard deviation for the Gaussian distribution.
#     :return: Array of heatmaps, one for each batch.
#     """
#     batch_size, num_points, _ = joints_2d.shape
#     heatmaps = np.zeros((batch_size, num_points, heatmap_size[0], heatmap_size[1]))

#     x_axis = np.linspace(0, heatmap_size[0] - 1, heatmap_size[0])
#     y_axis = np.linspace(0, heatmap_size[1] - 1, heatmap_size[1])
#     X, Y = np.meshgrid(x_axis, y_axis)

#     for i in range(batch_size):
#         for j in range(num_points):
#             x, y = joints_2d[i, j]
#             x = np.clip(x, 0, 1)
#             y = np.clip(y, 0, 1)
#             x_scaled = int(x * (heatmap_size[0] - 1))
#             y_scaled = int(y * (heatmap_size[1] - 1))

#             gaussian = np.exp(-((X - x_scaled)**2 + (Y - y_scaled)**2) / (2 * sigma**2))
#             normalized_gaussian = gaussian / gaussian.max()
#             heatmaps[i, j] = normalized_gaussian

#     return heatmaps



############################### hand kinematic functions ################################

class HandKinematic():
    def __init__(self, device) -> None:
        self.device = device
        self.SNAP_PARENT = [
            0,  # 0's parent
            0,  # 1's parent
            1,
            2,
            3,
            0,  # 5's parent
            5,
            6,
            7,
            0,  # 9's parent
            9,
            10,
            11,
            0,  # 13's parent
            13,
            14,
            15,
            0,  # 17's parent
            17,
            18,
            19,
        ]
        self.JOINT_ROOT_IDX = 9
        self.REF_BONE_LINK = (0, 9)  # mid mcp
        # bone indexes in 20 bones setting
        self.ID_ROOT_bone = [0, 4, 8, 12, 16] # ROOT_bone from wrist to MCP
        self.ID_PIP_bone = [1, 5, 9, 13, 17]  # PIP_bone from MCP to PIP
        self.ID_DIP_bone = [2, 6, 10, 14, 18]  # DIP_bone from  PIP to DIP
        self.ID_TIP_bone = [3, 7, 11, 15, 19]  # TIP_bone from DIP to TIP
        self.epsilon_float32 = torch.finfo(torch.float32).eps
        
    def __angle_between(self, v1, v2):
        """
        Calculate the angle between two vectors (tensor) with shape (batch, 3)
        """
        # v1_u = F.normalize(v1, dim=-1)
        # v2_u = F.normalize(v2, dim=-1)
        epsilon = 1e-7
        cos = F.cosine_similarity(v1, v2, dim=-1).clamp(-1 + epsilon, 1 - epsilon)  
        theta = torch.acos(cos)  # (B)
        return theta

    def __normalize(self, v):
        """
        Normalize a tensor with shape (batch, 3)

        """
        return F.normalize(v, dim=-1)
    
    def __axangle2mat(delf, axis, angle, is_normalized=False):
        '''
        input axis-angle representations of rotations, normalizes the axis vectors if necessary,
        and computes the corresponding rotation matrices using trigonometric functions.
        :param axis: (B,...,3)
        :param angle: (B,...,)
        :param is_normalized:
        :return: B,...,*3*3
        '''
        batch_size = axis.shape[0]
        
        if not is_normalized:
            axis = F.normalize(axis, dim=-1)
        x = axis[..., 0]
        y = axis[..., 1]
        z = axis[..., 2]
        
        c = torch.cos(angle)
        s = torch.sin(angle)
        t = 1 - c
        return torch.stack([t*x*x+c, t*x*y-s*z, t*x*z+s*y,
                            t*x*y+s*z, t*y*y+c, t*y*z-s*x,
                            t*x*z-s*y, t*y*z+s*x, t*z*z+c], dim=-1).view(batch_size, -1, 3, 3)

    def __veclen(self, v):
        # the vector length of a tensor with shape (batch, 3)
        return torch.sqrt((v*v).sum(-1))
    
    def __get_rotation_matrix(self, angles, device):
        #angles:batch*3
        alpha, beta, gamma = angles[:,0],angles[:,1],angles[:,2]
        w_x = torch.tensor([[0,0,0],[0,0,-1],[0,1,0]]).repeat(alpha.shape[0],1,1).float().to(device)
        w_y = torch.tensor([[0,0,1],[0,0,0],[-1,0,0]]).repeat(alpha.shape[0],1,1).float().to(device)
        w_z = torch.tensor([[0,-1,0],[1,0,0],[0,0,0]]).repeat(alpha.shape[0],1,1).float().to(device)
        I = torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).repeat(alpha.shape[0],1,1).float().to(device)
        r_x = I + torch.sin(alpha).resize(alpha.shape[0],1,1) *w_x+ (1-torch.cos(alpha)).resize(alpha.shape[0],1,1) *torch.matmul(w_x,w_x)
        r_y = I +torch.sin(beta).resize(alpha.shape[0],1,1)*w_y+(1-torch.cos(beta)).resize(alpha.shape[0],1,1)*torch.matmul(w_y,w_y)
        r_z = I +torch.sin(gamma).resize(alpha.shape[0],1,1)*w_z+(1-torch.cos(gamma)).resize(alpha.shape[0],1,1)*torch.matmul(w_z,w_z)
        return torch.matmul(r_y, torch.matmul(r_z,r_x))

    def __calculate_joint_position(self, prev_joint, length, prev_rotation, angles, device):
        local_vector = np.array([length, 0, 0])
        local_vector = torch.tensor(local_vector).repeat(angles.shape[0],1).float().to(device)
        current_rotation = torch.matmul(prev_rotation,self.__get_rotation_matrix(angles, device=device))
        global_vector = torch.matmul(current_rotation, local_vector.unsqueeze(-1)).squeeze(-1).to(device)
        new_joint = prev_joint + global_vector
        return new_joint, current_rotation

    def __FK_Finger(self, joints,index,plam_angle,plam_length,finger_lenth,input_26Dof, device):
        #input_26Dof: batch*26, type:torch.tensor
        # wrist_angle: batch*1*3
        wrist_angle = input_26Dof[:,3:6]
        batch_size = wrist_angle.shape[0]
        # transform wrist_angle from batch*3 to batch*1*3
        wrist_angle = wrist_angle.unsqueeze(1).to(device)

        palm_angle = [[0,0,plam_angle[index]]]
        palm_angle = torch.tensor(palm_angle).repeat(input_26Dof.shape[0],1,1).float().to(device)
        # Define a single finger with 4 DoF (3 joints)
        finger_lengths = [0,plam_length[index]]+finger_lenth[index]
        angle_from_26dof = input_26Dof[:,6+index*4:6+(index+1)*4]
        angle_from_26dof = angle_from_26dof.unsqueeze(1).to(device)
        
        finger_angles = torch.cat((wrist_angle,palm_angle.to(device),torch.cat((
            torch.zeros(batch_size,1, 1, device=device),  
            angle_from_26dof[:,:, 1].unsqueeze(-1),
            angle_from_26dof[:,:, 0].unsqueeze(-1)
        ), dim=-1).to(device),
        torch.cat((
            torch.zeros(batch_size,1, 1, device=device),
            angle_from_26dof[:,:, 2].unsqueeze(-1),
            torch.zeros(batch_size, 1,1, device=device)
        ), dim=-1).to(device),
        torch.cat((
            torch.zeros(batch_size, 1,1, device=device),
            angle_from_26dof[:,:, 3].unsqueeze(-1),
            torch.zeros(batch_size, 1,1, device=device)
        ), dim=-1).to(device)),1)

        # Initialize the first joint position at the palm and its rotation matrix
        prev_joint = torch.clone(joints[:,0,:3]).to(device)
        prev_rotation = np.identity(3)
        prev_rotation = torch.tensor(prev_rotation).repeat(batch_size,1,1).float().to(device)
        for i in range(5):
            new_joint, new_rotation = self.__calculate_joint_position(prev_joint, finger_lengths[i], prev_rotation, finger_angles[:,i,:], device)
            if i!=0:
                joints[:,index*4+i,:] = new_joint
            prev_joint = new_joint
            prev_rotation = new_rotation
        return joints

    def forward_kinematic(self, input_26Dof=None, bone_lenght=None):
        '''
        input_26Dof: batch*26, type:torch.tensor
            input_26Dof[:3] is the root position
            input_26Dof[3:6] is the root rotation
            input_26Dof[6:10] is the thumb
            input_26Dof[10:14] is the index finger
            input_26Dof[14:18] is the middle finger
            input_26Dof[18:22] is the ring finger
            input_26Dof[22:26] is the pinky finger
        output: batch*21*3, type:torch.tensor
        '''
        device = self.device
        batch_size = len(input_26Dof)
        if input_26Dof is None:
            input_26Dof = torch.zeros(batch_size, 26,requires_grad=True).to(device)
        left_plam_angle = [np.arccos(5/6.4),np.arccos(10/10.19),0,-np.arccos(10/10.19),-np.arccos(10/10.77)]
        right_plam_angle = [-np.arccos(5/6.4),-np.arccos(10/10.19),0,np.arccos(10/10.19),np.arccos(10/10.77)]
        
        """
        The parameters of the bone length of the 5 fingers from paper: 
            @article{dragulescu3DActiveWorkspace2007,
            title = {{{3D}} Active Workspace of Human Hand Anatomical Model},
            author = {Dragulescu, Doina and Perdereau, V{\'e}ronique and Drouin, Michel and Ungureanu, Loredana and Menyhardt, Karoly},
            year = {2007},
            month = may,
            journal = {BioMedical Engineering OnLine},
            volume = {6},
            number = {1},
            pages = {15},
            issn = {1475-925X},
            doi = {10.1186/1475-925X-6-15},
            urldate = {2023-08-29}
            }
        """
        # plam_length = [0.064,0.1019,0.100,0.1019,0.1077]
        # finger_length = [[0.025,0.015,0],
        #                 [0.025,0.015,0.010],
        #                 [0.030,0.015,0.015],
        #                 [0.025,0.015,0.010],
        #                 [0.025,0.015,0.010]]
        
        """ The mean and std of the bone length of the 5 fingers from leap motion
        thumb_length_list [0.07448986 0.02900618 0.01533575 0.        ] [1.80411242e-16 7.66747776e-16 3.15719673e-16 0.00000000e+00]
        index_finger_length_list [0.09570047 0.03651909 0.02049054 0.01111496] [2.15105711e-15 1.31838984e-15 6.07153217e-16 2.34187669e-16]
        middle_finger_length_list [0.09266819 0.04165332 0.02459831 0.01252055] [4.62130334e-15 1.14491749e-15 5.30825384e-16 5.30825384e-16]
        ring_finger_length_list [0.08683175 0.03715396 0.02282162 0.01179498] [1.40165657e-15 7.91033905e-16 1.11716192e-15 6.19296281e-16]
        pinky_finger_length_list [0.08405809 0.02920749 0.01584632 0.01080172] [2.63677968e-15 1.12410081e-15 9.08995101e-16 5.48172618e-16]
        
        thumb_length_list [0.07448986 0.02900618 0.01533575 0.        ] [2.08583151e-14 3.26128013e-15 3.48679419e-16 0.00000000e+00]
        index_finger_length_list [0.09570047 0.03651909 0.02049054 0.01111496] [1.67921232e-14 4.21190860e-15 9.05525654e-16 1.52829138e-15]
        middle_finger_length_list [0.09266819 0.04165332 0.02459831 0.01252055] [1.81105131e-14 3.53883589e-15 4.59007832e-15 1.45369827e-15]
        ring_finger_length_list [0.08683175 0.03715396 0.02282162 0.01179498] [2.02199368e-14 4.22578639e-15 3.10168558e-15 2.29850861e-15]
        pinky_finger_length_list [0.08405809 0.02920749 0.01584632 0.01080172] [3.81639165e-15 2.92127433e-15 3.06005221e-15 7.33788030e-16]
        """  
        if bone_lenght is not None:
            plam_length = bone_lenght['plam_length']
            finger_length = bone_lenght['finger_length']
        plam_length = [0.07448986,0.09570047,0.09266819,0.08683175,0.08405809]
        finger_length = [[0.02900618,0.01533575,0.],
                        [0.03651909,0.02049054,0.01111496],
                        [0.04165332,0.02459831,0.01252055],
                        [0.03715396,0.02282162,0.01179498],
                        [0.02920749,0.01584632,0.01080172]]
        
        left_joints = torch.empty(batch_size, 21, 3).to(device)   # if this is the left hand
        left_joints[:,0,:] = input_26Dof[:,0:3]
        for index in range(5):
            self.__FK_Finger(left_joints,index,left_plam_angle,plam_length,finger_length,input_26Dof, device)
        right_joints = torch.empty(batch_size, 21, 3).to(device)  # if this is the right hand
        right_joints[:,0,:] = input_26Dof[:,0:3]
        for index in range(5):
            self.__FK_Finger(right_joints,index,right_plam_angle,plam_length,finger_length,input_26Dof, device)
        return left_joints, right_joints
    
    def inverse_kinematic(self, joints_position):
        """
        joints_position; tensor with shape (batch, 21, 3)
        reture the root position, wrist orientation and the joint orientation (26 Dof)
        """
        root_point_global_position = joints_position[:, 0].clone()   # shape (batch, 3)
        all_bones = joints_position - joints_position[:, self.SNAP_PARENT, :]  # shape (batch, 21, 3) the bone vector of each joint
        all_bones = all_bones[:,1:] # shape (batch, 20, 3) the bone vector of each joint, the writst is not included
        root_bones = all_bones[:, self.ID_ROOT_bone] # shape (batch, 5, 3) the bone vector from wrist to MCP
        
        # caculating the rotation representation of wrist orientation using Euler angles with order: x, y, z 
        wrist_ori = self.__normalize(root_bones[:, 2]) # shape (batch, 3) the wrist vector from wrist to middle finger MCP
        # Eular angle on z axis
        wrist_ori_y_0_x_positive = wrist_ori.clone()
        wrist_ori_y_0_x_positive[:, 1] = 0
        z_rotation = self.__angle_between(wrist_ori_y_0_x_positive, wrist_ori)/math.pi*180  # shape (batch, )
        direction_vector = torch.tensor([1., 0., 0.], device=self.device)  # Ensure the device matches
        expanded_direction_vector = direction_vector.unsqueeze(0).expand(wrist_ori.size(0), -1)
        full_cross_product = torch.cross(wrist_ori, expanded_direction_vector, dim=1)
        z_operate = full_cross_product[:, 2]
        z_rotation = torch.where(z_operate>0, -z_rotation, z_rotation)
        z_rotation = torch.where((wrist_ori[:, 0] < 0) & (wrist_ori[:, 1] < 0), -180 - z_rotation, z_rotation)
        z_rotation = torch.where((wrist_ori[:, 0] < 0) & (wrist_ori[:, 1] > 0), 180 - z_rotation, z_rotation)
        
        # Eular angle on y axis
        wrist_ori_y_0_x_positive = wrist_ori.clone()
        wrist_ori_y_0_x_positive[:, 1] = 0
        wrist_ori_y_0_x_positive[:, 0] = torch.abs(wrist_ori_y_0_x_positive[:, 0].clone())
        y_temp = torch.tensor([1.,0.,0.]).expand(wrist_ori_y_0_x_positive.shape[0], 3).to(self.device)
        y_rotation = self.__angle_between(wrist_ori_y_0_x_positive, y_temp)/math.pi*180  # shape (batch, )
        # if the z component of wrist_ori is negative, then the y_rotation is negative
        y_rotation = torch.where(wrist_ori_y_0_x_positive[:, 2]<0, -y_rotation, y_rotation)
        
        # Eular angle on x axis
        w_y = torch.tensor([[0.,0.,1.],
                            [0.,0.,0.],
                            [-1.,0.,0.]]).expand(wrist_ori.shape[0], 3, 3).to(self.device)  # shape (batch, 3, 3)
        w_z = torch.tensor([[0.,-1,0.],
                            [1.,0.,0.],
                            [0.,0.,0.]]).expand(wrist_ori.shape[0], 3, 3).to(self.device)
        I = torch.tensor([[1.,0.,0.],
                            [0.,1.,0.],
                            [0.,0.,1.]]).expand(wrist_ori.shape[0], 3, 3).to(self.device)
        r_y = I + torch.sin(y_rotation/180*math.pi).unsqueeze(-1).unsqueeze(-1)*w_y + (1-torch.cos(y_rotation/180*math.pi)).unsqueeze(-1).unsqueeze(-1)*torch.matmul(w_y, w_y)
        r_z = I + torch.sin(z_rotation/180*math.pi).unsqueeze(-1).unsqueeze(-1)*w_z + (1-torch.cos(z_rotation/180*math.pi)).unsqueeze(-1).unsqueeze(-1)*torch.matmul(w_z, w_z)
        ver_t = torch.matmul(r_y, r_z)
        ver_ty = torch.sum(torch.mul(ver_t, torch.tensor([0.,1.,0.]).expand(wrist_ori.shape[0], 1,3).to(self.device)), dim = -1)
        ver_tz = torch.sum(torch.mul(ver_t, torch.tensor([0.,0.,1.]).expand(wrist_ori.shape[0], 1, 3).to(self.device)), dim = -1)
        ver = joints_position[:,5] - joints_position[:,9]
        x1 = self.__angle_between(ver,ver_ty)/math.pi*180
        x2 = self.__angle_between(ver,ver_tz)/math.pi*180
        condition = (torch.abs(x1 + x2 - 90) < 0.01) | ((x1 > 90) & (x2 < 90))
        x_rotation = torch.where(condition, x1, -x1)
        wrist_ori = torch.stack([x_rotation, y_rotation, z_rotation], dim=-1)
        
        # caculating the rotation representation of joint orientation using axis-angle representation
        PIP_bones = all_bones[:, self.ID_PIP_bone].clone()  # shape (batch, 5, 3)
        DIP_bones = all_bones[:, self.ID_DIP_bone].clone() # shape (batch, 5, 3)
        TIP_bones = all_bones[:, self.ID_TIP_bone].clone() # shape (batch, 5, 3)
        
        # using the parent-bone as the z axis for all PIP, DIP and TIP
        ALL_Z_axis = self.__normalize(all_bones)      # be unit vector
        PIP_Z_axis = ALL_Z_axis[:, self.ID_ROOT_bone]
        DIP_Z_axis = ALL_Z_axis[:, self.ID_PIP_bone]
        TIP_Z_axis = ALL_Z_axis[:, self.ID_DIP_bone]
        
        # calculating the x-axis and y-axis of the PIP
        local_normals = self.__normalize(torch.cross(root_bones[:,1:5], root_bones[:,0:4])) # shape (batch, 4, 3)
        PIP_X_axis = torch.zeros(PIP_Z_axis.shape).to(self.device)  # shape (batch, 5, 3)
        PIP_X_axis[:,[0, 1, 4], :] = -local_normals[:, [0, 1, 3], :] # shape (batch, 3, 3), the x-axis of the thumb, ring and pinky
        PIP_X_axis[:,2:4] = -self.__normalize(local_normals[:,2:4] + local_normals[:,1:3]) # shape (batch, 2, 3), the x-axis of the index and middle
        PIP_Y_axis = self.__normalize(torch.cross(PIP_Z_axis, PIP_X_axis)) # shape (batch, 5, 3)
        PIP_X_axis = self.__normalize(PIP_X_axis)
        # calculating the rotation representation of PIP orientation using axis-angle representation
        tmp = torch.sum(PIP_bones.clone() * PIP_Y_axis.clone(), dim=-1, keepdim=True)
        PIP_bones_xz = PIP_bones - tmp * PIP_Y_axis
        PIP_theta_flexion = self.__angle_between(PIP_bones_xz, PIP_Z_axis)
        PIP_theta_abduction = self.__angle_between(PIP_bones_xz, PIP_bones)
        # x-component of the bone vector
        tmp = torch.sum((PIP_bones * PIP_X_axis), dim=-1)
        tmp_index = torch.where(tmp < 0)
        PIP_theta_flexion[tmp_index] = -PIP_theta_flexion[tmp_index]
        # y-component of the bone vector
        tmp = torch.sum((PIP_bones * PIP_Y_axis), dim=-1)
        tmp_index = torch.where(tmp < 0)
        PIP_theta_abduction[tmp_index] = -PIP_theta_abduction[tmp_index]

        # calculating the rotation matrix that transform the PIP orientation to the DIP orientation, so that the x-axis of the DIP is identical to the x-axis of the PIP
        temp_axis = self.__normalize(torch.cross(PIP_Z_axis, PIP_bones))
        temp_alpha = self.__angle_between(PIP_Z_axis, PIP_bones)
        temp_R = self.__axangle2mat(axis=temp_axis, angle=temp_alpha, is_normalized=True)
        
        # calculating the rotation representation of DIP orientation using axis-angle representation
        DIP_X_axis = torch.matmul(temp_R, PIP_X_axis[:, :, :, np.newaxis])
        DIP_Y_axis = torch.matmul(temp_R, PIP_Y_axis[:, :, :, np.newaxis])
        DIP_X_axis = DIP_X_axis.squeeze(-1)
        DIP_Y_axis = DIP_Y_axis.squeeze(-1)
        tmp = torch.sum(DIP_bones.clone() * DIP_Y_axis.clone(), dim=-1, keepdim=True)
        DIP_bones_xz = DIP_bones - tmp * DIP_Y_axis
        # DIP_bones_xz_clip = torch.clamp(DIP_bones_xz, min=-1e6, max=1e6)
        DIP_theta_flexion = self.__angle_between(DIP_bones_xz, DIP_Z_axis)
        DIP_theta_abduction = self.__angle_between(DIP_bones_xz, DIP_bones)
        # x-component of the bone vector
        tmp = torch.sum((DIP_bones * DIP_X_axis), dim=-1)
        tmp_index = torch.where(tmp < 0)
        DIP_theta_flexion[tmp_index] = -DIP_theta_flexion[tmp_index]
        # y-component of the bone vector
        tmp = torch.sum((DIP_bones * DIP_Y_axis), dim=-1)
        tmp_index = torch.where(tmp < 0)
        DIP_theta_abduction[tmp_index] = -DIP_theta_abduction[tmp_index]

        # calculating the rotation matrix that transform the DIP orientation to the TIP orientation, so that the x-axis of the TIP is identical to the x-axis of the DIP
        temp_axis = self.__normalize(torch.cross(DIP_Z_axis, DIP_bones))
        temp_alpha = self.__angle_between(DIP_Z_axis, DIP_bones)
        temp_R = self.__axangle2mat(axis=temp_axis, angle=temp_alpha, is_normalized=True)

        TIP_X_axis = torch.matmul(temp_R, DIP_X_axis[:, :, :, np.newaxis])
        TIP_Y_axis = torch.matmul(temp_R, DIP_Y_axis[:, :, :, np.newaxis])
        TIP_X_axis = TIP_X_axis.squeeze(-1)
        TIP_Y_axis = TIP_Y_axis.squeeze(-1)
        tmp = torch.sum(TIP_bones.clone() * TIP_Y_axis.clone(), dim=-1, keepdim=True)
        # tmp2 = tmp * TIP_Y_axis
        # tmp2_clipped = torch.clamp(tmp2, min=-1e6, max=1e6)
        # TIP_bones_xz = TIP_bones.clone() - tmp2_clipped
        # TIP_bones_xz_clip = torch.clamp(TIP_bones_xz, min=-1e6, max=1e6)
        TIP_bones_xz = TIP_bones - tmp * TIP_Y_axis
        TIP_theta_flexion = self.__angle_between(TIP_bones_xz, TIP_Z_axis)
        TIP_theta_abduction = self.__angle_between(TIP_bones_xz, TIP_bones)

        # x-component of the bone vector
        tmp = torch.sum((TIP_bones * TIP_X_axis), dim=-1)
        tmp_index = torch.where(tmp < 0)
        TIP_theta_flexion[tmp_index] = -TIP_theta_flexion[tmp_index]
        # y-component of the bone vector
        tmp = torch.sum((TIP_bones * TIP_Y_axis), dim=-1)
        tmp_index = torch.where(tmp < 0)
        TIP_theta_abduction[tmp_index] = -TIP_theta_abduction[tmp_index]
    
        ALL_theta_flexion = torch.concatenate((PIP_theta_flexion, DIP_theta_flexion, TIP_theta_flexion), dim=1)
        ALL_theta_abduction = torch.concatenate((PIP_theta_abduction, DIP_theta_abduction, TIP_theta_abduction), dim=1)
        joints_ori = torch.stack((ALL_theta_flexion, ALL_theta_abduction), dim=2)

        return root_point_global_position, wrist_ori/180*math.pi, joints_ori
    

class HandRegulation():
    def __init__(self, device) -> None:        
        # Convert angle bounds to tensors
        self.device = device
        self.thumb_range = torch.tensor([
            [-30/180*math.pi, 30/180*math.pi],  
            [-25/180*math.pi, 80/180*math.pi], 
            [0/180*math.pi, 90/180*math.pi], 
            [0/180*math.pi, 90/180*math.pi], 
        ]).to(device)
        
        self.otherfinger_range = torch.tensor([
            [-30/180*math.pi, 30/180*math.pi],
            [-25/180*math.pi, 85/180*math.pi],  
            [0/180*math.pi, 90/180*math.pi], 
            [0/180*math.pi, 90/180*math.pi],
        ]).to(device)
        
        # self.thumb_range = [
        #     [-10/180*math.pi, 20/180*math.pi],  
        #     [-15/180*math.pi, 80/180*math.pi], 
        #     [0/180*math.pi, 90/180*math.pi], 
        #     [0/180*math.pi, 90/180*math.pi], 
        # ]
        
        # self.otherfinger_range = [
        #     [-10/180*math.pi, 20/180*math.pi],
        #     [-15/180*math.pi, 85/180*math.pi],  
        #     [0/180*math.pi, 90/180*math.pi], 
        #     [0/180*math.pi, 70/180*math.pi],
        # ]

    def forward(self, Dof26):
        # Expect Dof26 to be of shape (batch_size, 26)
        # No need to slice as in the numpy version since we're dealing with batches
        
        # Process each finger range using tensor operations
        for i in range(4):
            # Thumb
            thumb = Dof26[:, 6:10]
            thumb[:, i] = torch.clamp(thumb[:, i], self.thumb_range[i, 0], self.thumb_range[i, 1])
            # Other fingers
            for j in range(4):  # For each of the other fingers
                finger = Dof26[:, 10+4*j:14+4*j]  # Index, Middle, Ring, Pinky
                finger[:, i] = torch.clamp(finger[:, i], self.otherfinger_range[i, 0], self.otherfinger_range[i, 1])
        return Dof26
  
######################### hand structure loss functions #########################
class BoneLengthLoss(nn.Module):
    def __init__(self):
        super(BoneLengthLoss, self).__init__()
        self.HAND_CONNECTIONS = torch.tensor([
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ])

    def forward(self, pred, gt):
        # Extract joints based on connections for both pred and gt
        pred_joints1 = pred[:, self.HAND_CONNECTIONS[:, 0]]
        pred_joints2 = pred[:, self.HAND_CONNECTIONS[:, 1]]
        gt_joints1 = gt[:, self.HAND_CONNECTIONS[:, 0]]
        gt_joints2 = gt[:, self.HAND_CONNECTIONS[:, 1]]
        # Calculate bone lengths for both pred and gt
        pred_bone_lengths = torch.norm(pred_joints1 - pred_joints2, dim=2)
        gt_bone_lengths = torch.norm(gt_joints1 - gt_joints2, dim=2)
        # Calculate the mean square error of the bone lengths
        loss = torch.mean((pred_bone_lengths - gt_bone_lengths) ** 2)
        return loss


class KinematicChainLoss(nn.Module):
    def __init__(self, device):
        super(KinematicChainLoss, self).__init__()
        self.hand_kinematic = HandKinematic(device)

    def forward(self, pred, gt):
        # pred and gt are both batch of 3D hand pose, with shape (batch, 21, 3)
        # the 21 joints are in the order of the MANO model
        # first calculate the bone lengths of the prediction based on the hand connections
        pred_root_point_global_position, pred_wrist_ori, pred_joints_ori = self.hand_kinematic.inverse_kinematic(pred)
        gt_root_point_global_position, gt_wrist_ori, gt_joints_ori = self.hand_kinematic.inverse_kinematic(gt)
        
        loss_root = torch.mean((pred_root_point_global_position - gt_root_point_global_position) ** 2)
        loss_wrist = torch.mean((pred_wrist_ori - gt_wrist_ori) ** 2)
        loss_joints = torch.mean((pred_joints_ori - gt_joints_ori) ** 2)
        loss = loss_wrist + loss_joints
        return 0.01*loss
    


def SubpageInterpolating(subpage):
    shape = subpage.shape
    mat = subpage.copy()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mat[i,j] > 0.0:
                continue
            num = 0
            try:
                top = mat[i-1,j]
                num = num+1
            except:
                top = 0.0
            
            try:
                down = mat[i+1,j]
                num = num+1
            except:
                down = 0.0
            
            try:
                left = mat[i,j-1]
                num = num+1
            except:
                left = 0.0
            
            try:
                right = mat[i,j+1]
                num = num+1
            except:
                right = 0.0
            mat[i,j] = (top + down + left + right)/num
    return mat