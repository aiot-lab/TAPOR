import matplotlib.pyplot as plt
import numpy as np
# import torch
# import mano
# from mano.utils import Mesh
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# import config as cfg
import math
import numpy as np
import torch


class hand_model_config():
    def __init__(self):
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
        self.ID_ROOT_bone = np.array([0, 4, 8, 12, 16])  # ROOT_bone from wrist to MCP
        self.ID_PIP_bone = np.array([1, 5, 9, 13, 17])  # PIP_bone from MCP to PIP
        self.ID_DIP_bone = np.array([2, 6, 10, 14, 18])  # DIP_bone from  PIP to DIP
        self.ID_TIP_bone = np.array([3, 7, 11, 15, 19])  # TIP_bone from DIP to TIP
 
cfg = hand_model_config()
 

def angle_between(v1, v2):
    '''
    :param v1: B*3
    :param v2: B*3
    :return: B
    '''
    v1_u = normalize(v1.copy())
    v2_u = normalize(v2.copy())
    inner_product = np.sum(v1_u * v2_u, axis=-1)
    tmp = np.clip(inner_product, -1.0, 1.0)
    tmp = np.arccos(tmp)
    return tmp


def normalize(vec_):
    '''
    :param vec:  B*3
    :return:  B*1
    '''
    vec = vec_.copy()
    len = calcu_len(vec) + 1e-8
    return vec / len


def axangle2mat(axis, angle, is_normalized=False):
    '''
    :param axis: B*3
    :param angle: B*1
    :param is_normalized:
    :return: B*3*3
    '''
    if not is_normalized:
        axis = normalize(axis)
    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    Q = np.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])
    Q = Q.transpose(2, 0, 1)

    return Q


def calcu_len(vec):
    '''
    calculate length of vector
    :param vec: B*3
    :return: B*1
    '''
    return np.linalg.norm(vec, axis=-1, keepdims=True)


def caculate_ja(joint, vis=False):
    '''
    :param joint: 21*3
    :param vis:
    :return: 15*2
    '''
    wrist_position = joint[0]
    ALL_bones = np.array([
        joint[i] - joint[cfg.SNAP_PARENT[i]]
        for i in range(1, 21)
    ])
    ROOT_bones = ALL_bones[cfg.ID_ROOT_bone]  # FROM THUMB TO LITTLE FINGER
    WRIST_ORI = ROOT_bones[2]
    ver = joint[5]-joint[9]

    WRIST_ORI = normalize(WRIST_ORI)
    # x = angle_between([ver[0],ver[1],0],ver)/math.pi*180
    #x = np.arctan(WRIST_ORI[2]/WRIST_ORI[1])/math.pi*180
    # x = angle_between([0,WRIST_ORI[1],WRIST_ORI[2]],[0,1,0])/math.pi*180
    z = angle_between([WRIST_ORI[0],0,WRIST_ORI[2]],WRIST_ORI)/math.pi*180
    z_operate = np.cross([WRIST_ORI[0],WRIST_ORI[1]],[1,0])
    if z_operate>0:
        z = -z
    if (WRIST_ORI[0]<0 and WRIST_ORI[1]<0):
        z=-180-z
    if (WRIST_ORI[0]<0 and WRIST_ORI[1]>0):
        z = 180-z
        
    yy = -1
    if z<90 and z>-90:
        yy = 1

    y = angle_between([WRIST_ORI[0],0,WRIST_ORI[2]],[yy,0,0])/math.pi*180
    y_operate = np.cross([WRIST_ORI[0],WRIST_ORI[2]],[yy,0])
    if y_operate<0:
        y = -y
    
    w_y = np.array([[0,0,1],
                    [0,0,0],
                    [-1,0,0]])
    w_z = np.array([[0,-1,0],
                    [1,0,0],
                    [0,0,0]])
    I = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1]])
    r_y = I +math.sin(y/180*math.pi)*w_y+(1-math.cos(y/180*math.pi))*np.dot(w_y,w_y)
    r_z = I +math.sin(z/180*math.pi)*w_z+(1-math.cos(z/180*math.pi))*np.dot(w_z,w_z)
    ver_t = np.dot(r_y,r_z)
    ver_tx = np.dot(ver_t,np.array([0,1,0]))
    ver_tz = np.dot(ver_t,np.array([0,0,1]))
    x1 = angle_between(ver_tx,ver)/math.pi*180
    x2 = angle_between(ver_tz,ver)/math.pi*180
    # print(x1,x2)
    x_operate = -1
    if np.abs(x1+x2-90)<0.01 or (x1>90 and x2<90):
        x_operate = 1
    x = x1*x_operate

    WRIST_ORI=[x,y,z]

    ### above is wrist orientation

    PIP_bones = ALL_bones[cfg.ID_PIP_bone]
    DIP_bones = ALL_bones[cfg.ID_DIP_bone]
    TIP_bones = ALL_bones[cfg.ID_TIP_bone]

    ALL_Z_axis = normalize(ALL_bones)
    PIP_Z_axis = ALL_Z_axis[cfg.ID_ROOT_bone]
    DIP_Z_axis = ALL_Z_axis[cfg.ID_PIP_bone]
    TIP_Z_axis = ALL_Z_axis[cfg.ID_DIP_bone]

    normals = normalize(np.cross(ROOT_bones[1:5], ROOT_bones[0:4]))

    # ROOT bones
    PIP_X_axis = np.zeros([5, 3])  # (5,3)
    PIP_X_axis[[0, 1, 4], :] = -normals[[0, 1, 3], :]
    PIP_X_axis[2:4] = -normalize(normals[2:4] + normals[1:3])
    PIP_Y_axis = normalize(np.cross(PIP_Z_axis, PIP_X_axis))

    tmp = np.sum(PIP_bones * PIP_Y_axis, axis=-1, keepdims=True)
    PIP_bones_xz = PIP_bones - tmp * PIP_Y_axis
    PIP_theta_flexion = angle_between(PIP_bones_xz, PIP_Z_axis)  # in global coordinate
    PIP_theta_abduction = angle_between(PIP_bones_xz, PIP_bones)  # in global coordinate
    # x-component of the bone vector
    tmp = np.sum((PIP_bones * PIP_X_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    PIP_theta_flexion[tmp_index] = -PIP_theta_flexion[tmp_index]
    # y-component of the bone vector
    tmp = np.sum((PIP_bones * PIP_Y_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    PIP_theta_abduction[tmp_index] = -PIP_theta_abduction[tmp_index]

    temp_axis = normalize(np.cross(PIP_Z_axis, PIP_bones))
    temp_alpha = angle_between(PIP_Z_axis, PIP_bones)  # alpha belongs to [0, pi]
    temp_R = axangle2mat(axis=temp_axis, angle=temp_alpha, is_normalized=True)

    # DIP bones
    DIP_X_axis = np.matmul(temp_R, PIP_X_axis[:, :, np.newaxis])
    DIP_Y_axis = np.matmul(temp_R, PIP_Y_axis[:, :, np.newaxis])
    DIP_X_axis = np.squeeze(DIP_X_axis)
    DIP_Y_axis = np.squeeze(DIP_Y_axis)

    tmp = np.sum(DIP_bones * DIP_Y_axis, axis=-1, keepdims=True)
    DIP_bones_xz = DIP_bones - tmp * DIP_Y_axis
    DIP_theta_flexion = angle_between(DIP_bones_xz, DIP_Z_axis)  # in global coordinate
    DIP_theta_abduction = angle_between(DIP_bones_xz, DIP_bones)  # in global coordinate
    # x-component of the bone vector
    tmp = np.sum((DIP_bones * DIP_X_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    DIP_theta_flexion[tmp_index] = -DIP_theta_flexion[tmp_index]
    # y-component of the bone vector
    tmp = np.sum((DIP_bones * DIP_Y_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    DIP_theta_abduction[tmp_index] = -DIP_theta_abduction[tmp_index]

    temp_axis = normalize(np.cross(DIP_Z_axis, DIP_bones))
    temp_alpha = angle_between(DIP_Z_axis, DIP_bones)  # alpha belongs to [pi/2, pi]
    temp_R = axangle2mat(axis=temp_axis, angle=temp_alpha, is_normalized=True)

    # TIP bones
    TIP_X_axis = np.matmul(temp_R, DIP_X_axis[:, :, np.newaxis])
    TIP_Y_axis = np.matmul(temp_R, DIP_Y_axis[:, :, np.newaxis])
    TIP_X_axis = np.squeeze(TIP_X_axis)
    TIP_Y_axis = np.squeeze(TIP_Y_axis)

    tmp = np.sum(TIP_bones * TIP_Y_axis, axis=-1, keepdims=True)
    TIP_bones_xz = TIP_bones - tmp * TIP_Y_axis
    TIP_theta_flexion = angle_between(TIP_bones_xz, TIP_Z_axis)  # in global coordinate
    TIP_theta_abduction = angle_between(TIP_bones_xz, TIP_bones)  # in global coordinate
    # x-component of the bone vector
    tmp = np.sum((TIP_bones * TIP_X_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    TIP_theta_flexion[tmp_index] = -TIP_theta_flexion[tmp_index]
    # y-component of the bone vector
    tmp = np.sum((TIP_bones * TIP_Y_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    TIP_theta_abduction[tmp_index] = -TIP_theta_abduction[tmp_index]

    if vis:
        fig = plt.figure(figsize=[50, 50])
        ax = fig.add_axes(Axes3D(fig))
        #ax = fig.gca(projection='3d')
        plt.plot(joint[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], 0],
                 joint[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], 1],
                 joint[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], 2], 'yo', label='keypoint')

        plt.plot(joint[:5, 0], joint[:5, 1],
                 joint[:5, 2],
                 '--y', )
        # label='thumb')
        plt.plot(joint[[0, 5, 6, 7, 8, ], 0], joint[[0, 5, 6, 7, 8, ], 1],
                 joint[[0, 5, 6, 7, 8, ], 2],
                 '--y',
                 )
        plt.plot(joint[[0, 9, 10, 11, 12, ], 0], joint[[0, 9, 10, 11, 12], 1],
                 joint[[0, 9, 10, 11, 12], 2],
                 '--y',
                 )
        plt.plot(joint[[0, 13, 14, 15, 16], 0], joint[[0, 13, 14, 15, 16], 1],
                 joint[[0, 13, 14, 15, 16], 2],
                 '--y',
                 )
        plt.plot(joint[[0, 17, 18, 19, 20], 0], joint[[0, 17, 18, 19, 20], 1],
                 joint[[0, 17, 18, 19, 20], 2],
                 '--y',
                 )
        plt.plot(joint[4][0], joint[4][1], joint[4][2], 'rD', label='thumb')
        plt.plot(joint[8][0], joint[8][1], joint[8][2], 'r*', label='index')
        plt.plot(joint[12][0], joint[12][1], joint[12][2], 'r+', label='middle')
        plt.plot(joint[16][0], joint[16][1], joint[16][2], 'rx', label='ring')
        plt.plot(joint[20][0], joint[20][1], joint[20][2], 'ro', label='pinky')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        parent = np.array(cfg.SNAP_PARENT[1:])
        x, y, z = joint[parent, 0], joint[parent, 1], joint[parent, 2]
        u, v, w = ALL_bones[:, 0], ALL_bones[:, 1], ALL_bones[:, 2],
        ax.quiver(x, y, z, u, v, w, length=0.25, color="black", normalize=True)

        ALL_X_axis = np.stack((PIP_X_axis, DIP_X_axis, TIP_X_axis), axis=0).reshape(-1, 3)
        ALL_Y_axis = np.stack((PIP_Y_axis, DIP_Y_axis, TIP_Y_axis), axis=0).reshape(-1, 3)
        ALL_Z_axis = np.stack((PIP_Z_axis, DIP_Z_axis, TIP_Z_axis), axis=0).reshape(-1, 3)
        ALL_Bone_xz = np.stack((PIP_bones_xz, DIP_bones_xz, TIP_bones_xz), axis=0).reshape(-1, 3)

        ALL_joints_ID = np.array([cfg.ID_PIP_bone, cfg.ID_DIP_bone, cfg.ID_TIP_bone]).flatten()

        jx, jy, jz = joint[ALL_joints_ID, 0], joint[ALL_joints_ID, 1], joint[ALL_joints_ID, 2]
        ax.quiver(jx, jy, jz, ALL_X_axis[:, 0], ALL_X_axis[:, 1], ALL_X_axis[:, 2], length=0.05, color="r",
                  normalize=True)
        #print(ALL_X_axis)
        ax.quiver(jx, jy, jz, ALL_Y_axis[:, 0], ALL_Y_axis[:, 1], ALL_Y_axis[:, 2], length=0.10, color="g",
                  normalize=True)
        ax.quiver(jx, jy, jz, ALL_Z_axis[:, 0], ALL_Z_axis[:, 1], ALL_Z_axis[:, 2], length=0.10, color="b",
                  normalize=True)
        ax.quiver(jx, jy, jz, ALL_Bone_xz[:, 0], ALL_Bone_xz[:, 1], ALL_Bone_xz[:, 2], length=0.25, color="pink",
                  normalize=True)

        plt.legend()
        # ax.view_init(-180, 90)
        plt.show()

    ALL_theta_flexion = np.stack((PIP_theta_flexion, DIP_theta_flexion, TIP_theta_flexion), axis=0).flatten()  # (15,)
    ALL_theta_abduction = np.stack((PIP_theta_abduction, DIP_theta_abduction, TIP_theta_abduction),
                                   axis=0).flatten()  # (15,)
    ALL_theta = np.stack((ALL_theta_flexion, ALL_theta_abduction), axis=1)  # (15, 2)

    return  WRIST_ORI,ALL_theta



if __name__ == '__main__':
    # Index joint 0: [0, 0, 0]
    # Index joint 1: [ 5.         -3.93430389 -0.69372393]
    # Index joint 2: [ 6.953125   -5.47114135 -0.96470984]
    # Index joint 3: [ 8.125      -6.39324382 -1.12730138]
    # Index joint 4: [ 8.125      -6.39324382 -1.12730138]
    # Index joint 5: [10.         -1.92884096 -0.3401067 ]
    # Index joint 6: [12.45338567 -2.40206003 -0.42354799]
    # Index joint 7: [13.92541708 -2.68599148 -0.47361277]
    # Index joint 8: [14.90677134 -2.87527911 -0.50698928]
    # Index joint 9: [10.  0.  0.]
    # Index joint 10: [13.  0.  0.]
    # Index joint 11: [14.5  0.   0. ]
    # Index joint 12: [16.  0.  0.]
    # Index joint 13: [10.          1.92884096  0.3401067 ]
    # Index joint 14: [12.45338567  2.40206003  0.42354799]
    # Index joint 15: [13.92541708  2.68599148  0.47361277]
    # Index joint 16: [14.90677134  2.87527911  0.50698928]
    # Index joint 17: [10.          3.9383569   0.69443858]
    # Index joint 18: [12.32126277  4.85255302  0.85563602]
    # Index joint 19: [13.71402043  5.4010707   0.95235449]
    # Index joint 20: [14.64252553  5.76674914  1.01683347]
    joint = np.array([[0.4124189104564952, 0.40874527453789955, 0.3620000183582306], [0.4823609280292961, 0.37056428154961585, 0.33912179060280323], [0.5174451560079001, 0.31606558786446026, 0.336362449452281], [0.5391198662447523, 0.27196909925343815, 0.33324234187602997], [0.5689850340984239, 0.24502062368081706, 0.33119024336338043], [0.4141571505460404, 0.25657176536073667, 0.37072162609547377], [0.42296099579322494, 0.2029850717814242, 0.3607007055543363], [0.42661183608116543, 0.16990233691474663, 0.3462239857763052], [0.426527017419137, 0.1417274203211891, 0.33483712002635], [0.36099103895126833, 0.25929343133489524, 0.368576318025589], [0.34061809413909133, 0.19922137701343962, 0.36169745403458364], [0.32913783398534313, 0.16242616716638691, 0.3491895869374275], [0.31844449189240015, 0.13379207177430677, 0.33967252634465694], [0.31919924395947913, 0.27233960304495763, 0.36102872353512794], [0.2889067917248591, 0.21952360533307147, 0.3505707122385502], [0.2749814692979956, 0.1856998928715209, 0.3392571546137333], [0.26307469062029265, 0.157911868602228, 0.3316231481730938], [0.2844426751312543, 0.29388415111944827, 0.349967653863132], [0.2533776646544568, 0.2516972189578465, 0.33847846277058125], [0.23490761918542427, 0.22464318504073752, 0.33387273363769054], [0.22116457072301007, 0.19999744076258846, 0.3315896615386009]])
    print(joint.shape)
    jj = [joint[0][0],joint[0][1],joint[0][2]]
    #print(joint[9]-joint[0])
    for j in range(21):
        joint[j] -= jj
    print(joint)
    Wrist,result = caculate_ja(joint, vis=False)#[up,out]
    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i][j] = result[i][j]/math.pi*180
    print(Wrist)
    
    result = result#left:up-down,right:out
    Dof26 = jj+Wrist
    print(result)
    for x in range(5):
        tmp = [result[x+5][0],result[x+10][0]]
        Dof26+=result[x].tolist()+tmp
    print(Dof26)
    #print(result)
