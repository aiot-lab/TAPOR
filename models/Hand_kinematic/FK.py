#26DoF:[6+4*5]
#lenth:mano format
#plam:L1->L4(Dragulescu, Doina, Véronique Perdereau, Michel Drouin, Loredana Ungureanu和Karoly Menyhardt. 
# 《3D active workspace of human hand anatomical model》. BioMedical Engineering OnLine 6, 期 1 (2007年5月2日): 
# 15. https://doi.org/10.1186/1475-925X-6-15.)


import numpy as np
import math
import matplotlib.pyplot as plt
import torch

############# input example: #############
# plam_angle = [-np.arccos(5/6.4),-np.arccos(10/10.19),0,np.arccos(10/10.19),np.arccos(10/10.77)]
# plam_length = [6.4,10.19,10,10.19,10.77]
# finger_length = [[2.5,1.5,0],
#                  [2.5,1.5,1],
#                  [3,1.5,1.5],
#                  [2.5,1.5,1],
#                  [2.5,1.5,1]]
# input_26Dof = [0]*26
# input_26Dof[3] = 20
# input_26Dof[6]=10
# lines = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]

def get_rotation_matrix(angles, device):
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

def calculate_joint_position(prev_joint, length, prev_rotation, angles, device):
    local_vector = np.array([length, 0, 0])
    local_vector = torch.tensor(local_vector).repeat(angles.shape[0],1).float().to(device)
    current_rotation = torch.matmul(prev_rotation,get_rotation_matrix(angles, device=device))
    global_vector = torch.matmul(current_rotation, local_vector.unsqueeze(-1)).squeeze(-1).to(device)
    new_joint = prev_joint + global_vector
    return new_joint, current_rotation

# Initialize some arbitrary joint angles and lengths for demonstration
def FK_Finger(joints,index,plam_angle,plam_length,finger_lenth,input_26Dof, device):
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
        new_joint, new_rotation = calculate_joint_position(prev_joint, finger_lengths[i], prev_rotation, finger_angles[:,i,:], device)
        if i!=0:
            joints[:,index*4+i,:] = new_joint
        prev_joint = new_joint
        prev_rotation = new_rotation
    # Output the positions of all joints
    return joints


def FK_cal(input_26Dof=None, device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
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
    # fingure_lenth = [[0.025,0.015,0],
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
    plam_length = [0.07448986,0.09570047,0.09266819,0.08683175,0.08405809]
    fingure_lenth = [[0.02900618,0.01533575,0.],
                    [0.03651909,0.02049054,0.01111496],
                    [0.04165332,0.02459831,0.01252055],
                    [0.03715396,0.02282162,0.01179498],
                    [0.02920749,0.01584632,0.01080172]]
    
    
    left_joints = torch.empty(batch_size, 21, 3).to(device)
    left_joints[:,0,:] = input_26Dof[:,0:3]
    for index in range(5):
        FK_Finger(left_joints,index,left_plam_angle,plam_length,fingure_lenth,input_26Dof, device)
    right_joints = torch.empty(batch_size, 21, 3).to(device)
    right_joints[:,0,:] = input_26Dof[:,0:3]
    for index in range(5):
        FK_Finger(right_joints,index,right_plam_angle,plam_length,fingure_lenth,input_26Dof, device)
    return left_joints, right_joints

