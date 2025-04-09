import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import random
import pickle   
from utils import generate_heatmaps
from tqdm import tqdm

def data_loader(path, file_names):
    # Load data from file
    maps = []
    ambient_temperatures = []
    l_depth_maps = []
    l_2d_joints = []
    l_3d_joints = []
    l_hand_depths = []
    l_left_right_flags = [] # indicate whether the hand is left or right, right: [0,1], left: [1,0]; for one of the baseline model
    
    skip_the_start_frames = 20
    
    for file_name in file_names:
        with open(os.path.join(path, file_name), 'rb') as f:
            data = pickle.load(f)
        maps.append(np.array(data['thermal_array_temperature_maps'][skip_the_start_frames:]))
        ambient_temperatures.append(np.array(data['thermal_array_ambient_temperatures'][skip_the_start_frames:]))
        l_depth_maps.append(np.array(data['label_depth_maps'][skip_the_start_frames:]))
        l_2d_joints+=data['label_2d_joints'][skip_the_start_frames:]
        l_3d_joints+=data['label_3d_joints'][skip_the_start_frames:]
        l_hand_depths+=data['label_hand_depth'][skip_the_start_frames:]
        if 'L' in file_name:
            l_left_right_flags+= [[1,0] for _ in range(len(data['thermal_array_temperature_maps'][skip_the_start_frames:]))]
        else:
            l_left_right_flags+= [[0,1] for _ in range(len(data['thermal_array_temperature_maps'][skip_the_start_frames:]))]
    return np.concatenate(maps), np.concatenate(ambient_temperatures), np.concatenate(l_depth_maps), l_2d_joints, l_3d_joints, l_hand_depths, l_left_right_flags

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


def GetChessboard(shape):
    chessboard = np.indices(shape).sum(axis=0) % 2
    chessboard_inverse = np.where((chessboard==0)|(chessboard==1), chessboard^1, chessboard)
    return chessboard, chessboard_inverse

class Preprocess():
    def __init__(self):
        chessboard, _ = GetChessboard((24,32))
        self.chessboard = chessboard
        self.chessboard_inverse = np.where((chessboard==0)|(chessboard==1), chessboard^1, chessboard)
         
    def Outlier1TypeDelete(self, mat):
        subpage0 = mat * self.chessboard
        subpage1 = mat* self.chessboard_inverse
        num_pixels_subpage = int(np.sum(self.chessboard))
        if np.sum(subpage0) > 300*num_pixels_subpage or np.sum(subpage1) > 300*num_pixels_subpage:
            return 0     # we need to discard this sample
        return 1         # we can keep this sample
    
    def Outlier2TypeElimilate(self, mat):
        mat_copy = mat.copy()
        outlier_position = np.where(mat>300)
        rows = outlier_position[0]
        cols = outlier_position[1]
        stats = 1      # there is no outliers in this sample
        for index,row in enumerate(rows):
            i = row
            j = cols[index]
            num = 0
            stats = 2    # there exists outliers in this sample
            try:
                topleft = mat_copy[i-1,j-1]
                num = num+1
            except:
                topleft = 0.0
            
            try:
                bottomleft = mat_copy[i+1,j-1]
                num = num+1
            except:
                bottomleft = 0.0
            
            try:
                topright = mat_copy[i-1,j+1]
                num = num+1
            except:
                topright = 0.0
            
            try:
                bottomright = mat_copy[i+1,j+1]
                num = num+1
            except:
                bottomright = 0.0
            mat_copy[i,j] = (topleft + bottomleft + topright + bottomright)/num
        return mat_copy, stats
    
    def Forward(self, mat):
        """
        stats:
            0: discard this sample.
            1: all pixels are perfect.
            2: there are some outliers in this sample but fixed by interpolating.
        """
        mat = mat.copy()
        stats = self.Outlier1TypeDelete(mat)
        if stats:
            mat, stats = self.Outlier2TypeElimilate(mat)
        return mat, stats

class generate_heatmaps_class():
    def __init__(self, heatmap_size=(256, 256), sigma=10):
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.kernel = self._generate_guassian_kernel(np.array([[[0.5, 0.5]]]), heatmap_size=self.heatmap_size, sigma=self.sigma).squeeze()
    
    def _generate_guassian_kernel(self, joints_2d, heatmap_size=(256, 256), sigma=10):
        """
        Generate heatmaps for a batch of coordinate points using vectorized operations for speedup.
        :param joints_2d: Tensor of shape (batch_size, num_points, 2) containing normalized coordinates.
        :param heatmap_size: Size of the heatmap (width, height).
        :param sigma: Standard deviation for the Gaussian distribution.
        :return: Array of heatmaps, one for each batch.
        """
        batch_size, num_points, _ = joints_2d.shape
        heatmaps = np.zeros((batch_size, num_points, heatmap_size[0], heatmap_size[1]))

        x_axis = np.linspace(0, heatmap_size[0] - 1, heatmap_size[0])
        y_axis = np.linspace(0, heatmap_size[1] - 1, heatmap_size[1])
        X, Y = np.meshgrid(x_axis, y_axis)
        for i in range(batch_size):
            for j in range(num_points):
                x, y = joints_2d[i, j]
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                x_scaled = int(x * (heatmap_size[0] - 1))
                y_scaled = int(y * (heatmap_size[1] - 1))

                gaussian = np.exp(-((X - x_scaled)**2 + (Y - y_scaled)**2) / (2 * sigma**2))
                normalized_gaussian = gaussian / gaussian.max()
                heatmaps[i, j] = normalized_gaussian
        return heatmaps

    def __call__(self, joints_2d):
        """
        Generate heatmaps for a batch of coordinate points.
        :param joints_2d: Tensor of shape (batch_size, num_points, 2) containing normalized coordinates.
        :param heatmap_size: Size of the heatmap (width, height).
        :param sigma: Standard deviation for the Gaussian distribution.
        :return: Array of heatmaps, one for each batch.
        """
        batch_size, num_points, _ = joints_2d.shape
        heatmaps = []
        for i in range(batch_size):
            tmp = []
            for j in range(num_points):
                x, y = joints_2d[i, j]
                x_shift = np.clip(x, 0, 1) - 0.5
                x_sift_pixel = int(x_shift*self.heatmap_size[0])
                y_shift = np.clip(y, 0, 1) - 0.5
                y_shift_pixel = int(y_shift*self.heatmap_size[1])
                heatmap = np.roll(self.kernel.copy(), y_shift_pixel, axis=0)
                heatmap = np.roll(heatmap, x_sift_pixel, axis=1)
                tmp.append(heatmap)  
            heatmaps.append(tmp)     
        return np.array(heatmaps)

class RandomOrthogonalRotation():
    def __init__(self):
        pass
    
    def __call__(self, thermal_map=None, depth_map=None, label_2d_joint=None, label_3d_joint=None):
        # select degree from {0, 90, -90, 180}
        degree = np.random.choice([0, 90, -90, 180])
        new_thermal_map = self.rotate_2D_map(thermal_map, degree)
        new_depth_map = self.rotate_2D_map(depth_map, degree)   
        new_label_2d_joint = self.rotate_2D_joints_label(label_2d_joint, degree)
        new_label_3d_joint = self.rotate_3D_joints_label(label_3d_joint, degree)
        return new_thermal_map, new_depth_map, new_label_2d_joint, new_label_3d_joint
    
    # rotate a 2D map with a given degree
    def rotate_2D_map(self, map, degree):
        if map is None:
            return None
        if degree == 0:
            return map
        elif degree == 90:
            return np.rot90(map)
        elif degree == -90:
            return np.rot90(map, 3)
        elif degree == 180:
            return np.rot90(map, 2)
        else:
            raise ValueError('degree should be 0, 90, -90, 180')
        
    # rotate a 2D map with a given degree
    def rotate_2D_joints_label(self,label_2d_joint, degree):
        if label_2d_joint is None:
            return None
        points = np.array(label_2d_joint)[:,:] #(x,y) == (horizontal, vertical, depth_from_depth_map)
        new_points = []
        for point in points:
            x, y, d = point
            if degree == 0:
                new_points.append([x, y, d])
            elif degree == 90:
                new_points.append([y, 1-x, d])
            elif degree == -90:
                new_points.append([1-y, x, d])
            elif degree == 180:
                new_points.append([1-x, 1-y, d])
            else:
                raise ValueError('degree should be 0, 90, -90, 180')
        return np.array(new_points)

    def rotate_3D_joints_label(self, label_3d_joint, degree):
        if label_3d_joint is None:
            return None
        points = np.array(label_3d_joint) #(x,y,z) == (horizontal, vertical, depth)
        new_points = []
        for point in points:
            x, y, z = point
            if degree == 0:
                new_points.append([x, y, z])
            elif degree == 90:
                new_points.append([-y, x, z])
            elif degree == -90:
                new_points.append([y, -x, z])
            elif degree == 180:
                new_points.append([-x, -y, z])
            else:
                raise ValueError('degree should be 0, 90, -90, 180')
        return np.array(new_points)


class iHand_dataset(Dataset):
    def __init__(self, path, file_names, fragment_size = 1,map_size = (96,96), transform=None, require_heatmap=False, filter_out_no_hand=False, shuffle_fragment=False,sliding_window=False):
        super(iHand_dataset, self).__init__()
        """
        params:
            path: the path of the dataset
            file_names: the list of file names that used to construct the dataset from the path folder
            fragment_size: the number of thermal array maps in one sample, can be regarded as the channels of the input
            map_size: the size of the interpolated thermal map
            transform: the transform function // do not use it for now
            require_heatmap: whether to generate heatmap based on the 2d joints label which is used for the detection-based models
            filter_out_no_hand: whether to filter out the samples without hand
            shuffle_fragment: whether to shuffle the fragment, if true, the fragment will be shuffled every calling of the __getitem__ function
        """
        self.path = path
        self.sliding_window = sliding_window
        self.file_names = file_names
        if fragment_size >= 1:
            self.fragment_size = int(fragment_size)
        else:
            print("fragment_size should be an integer larger than 0")
            self.fragment_size = 1
        self.map_size = map_size
        if map_size[0] == 0 or map_size[1] == 0:
            self.map_size = (32,24) # keep as the original size
        self.transform = transform
        self.maps, self.ambient_temperatures, self.l_depth_maps, self.l_2d_joints, self.l_3d_joints, self.l_hand_depths, self.l_left_right_flags = data_loader(path, file_names)
        self.l_2d_joints_flag = []   # 0: no hand (2d label) in this frame, 1: hand in this frame
        self.l_3d_joints_flag = []   # 0: no hand (3d label) in this frame, 1: hand in this frame
        self.generate_2d_heatmap = require_heatmap 
        self.heatmap = []
        self.filter_out_no_hand = filter_out_no_hand
        self.sample_w_hand_index = []  # record the index of the samples with hand
        self.shuffle_fragment = shuffle_fragment   
        if self.filter_out_no_hand:
            for i in range(len(self.l_2d_joints)):
                if len(self.l_2d_joints[i]) == 0 or len(self.l_3d_joints[i]) == 0:
                    pass
                else:
                    self.sample_w_hand_index.append(i)
        
        if self.generate_2d_heatmap:
            self.heatmap_generator = generate_heatmaps_class(heatmap_size=self.map_size, sigma=5)
        
        for i in range(len(self.l_2d_joints)):
            if len(self.l_2d_joints[i]) == 0:
                self.l_2d_joints_flag.append(0)
                self.l_2d_joints[i] = np.zeros((21,3))  # 21 joints, the third dim is actually useless which is originally the depth of the joint get from the depth map
            else:
                self.l_2d_joints_flag.append(1)
        
        for i in range(len(self.l_3d_joints)):
            if len(self.l_3d_joints[i]) == 0:
                self.l_3d_joints_flag.append(0)
                self.l_3d_joints[i] = np.zeros((21,3))
            else:
                self.l_3d_joints_flag.append(1)
    
        # convert all to numpy array
        self.maps = np.array(self.maps)
        self.ambient_temperatures = np.array(self.ambient_temperatures)
        self.l_depth_maps = np.array(self.l_depth_maps)
        self.l_2d_joints = np.array(self.l_2d_joints)
        self.l_3d_joints = np.array(self.l_3d_joints)
        self.l_hand_depths = np.array(self.l_hand_depths)
        self.l_2d_joints_flag = np.array(self.l_2d_joints_flag)
        self.l_3d_joints_flag = np.array(self.l_3d_joints_flag)
        self.l_left_right_flags = np.array(self.l_left_right_flags)
        
        if self.filter_out_no_hand:
            # keep the samples with hand based on the sample_w_hand_index
            self.maps = self.maps[self.sample_w_hand_index]
            self.ambient_temperatures = self.ambient_temperatures[self.sample_w_hand_index]
            self.l_depth_maps = self.l_depth_maps[self.sample_w_hand_index]
            self.l_2d_joints = self.l_2d_joints[self.sample_w_hand_index]
            self.l_3d_joints = self.l_3d_joints[self.sample_w_hand_index]
            self.l_hand_depths = self.l_hand_depths[self.sample_w_hand_index]
            self.l_2d_joints_flag = self.l_2d_joints_flag[self.sample_w_hand_index]
            self.l_3d_joints_flag = self.l_3d_joints_flag[self.sample_w_hand_index]
            self.l_left_right_flags = self.l_left_right_flags[self.sample_w_hand_index]
        
        # preprocessing the data
        # interpolating the thermal map on the missing pixels due to the chessboard pattern reading of the sensor 
        # and resize the thermal map and depth map to the same size
        self.l_depth_maps = np.array([cv2.resize(depth_map, self.map_size) for depth_map in self.l_depth_maps])
        self.maps = np.array([cv2.resize(SubpageInterpolating(thermal_map), self.map_size, interpolation=cv2.INTER_NEAREST) for thermal_map in self.maps])
        # convert numpy.uint16 to numpy.float32
        self.l_depth_maps = self.l_depth_maps.astype(np.float32)
        self.maps = self.maps.astype(np.float32)
            
    def __len__(self):
        if self.shuffle_fragment:
            return int(len(self.maps)/self.fragment_size)-1  # -1 to avoid the last fragment which may be incomplete when we shuffle the fragment  
        if self.sliding_window:
            return int(len(self.maps)-self.fragment_size+1)
        return int(len(self.maps)/self.fragment_size)
    
    def __getitem__(self, idx):
        # get data
        if self.shuffle_fragment:
            idx_offset = random.randint(0, int(self.fragment_size)-1)
        else:
            idx_offset = 0
        if self.sliding_window:
            start_index = idx
            end_index = idx+self.fragment_size
        else:
            start_index = idx*self.fragment_size +idx_offset
            end_index = (idx+1)*self.fragment_size+idx_offset
        thermal_map = self.maps[start_index:end_index]
        ambient_temperature = self.ambient_temperatures[start_index:end_index]
        l_depth_map = self.l_depth_maps[start_index:end_index]
        l_2d_joint = self.l_2d_joints[start_index:end_index]
        l_2d_flag = self.l_2d_joints_flag[start_index:end_index]
        l_3d_joint = self.l_3d_joints[start_index:end_index]
        l_3d_flag = self.l_3d_joints_flag[start_index:end_index]
        l_hand_depth = self.l_hand_depths[start_index:end_index]
        l_left_right_flag = self.l_left_right_flags[start_index:end_index]
        
        # # Transform
        # if self.transform is not None:
        #     for j in range(len(thermal_map)):
        #         thermal_map[j], l_depth_map[j], l_2d_joint[j], l_3d_joint[j] = self.transform(thermal_map[j], l_depth_map[j], l_2d_joint[j], l_3d_joint[j]) 

        if self.generate_2d_heatmap:
            # self.heatmap = generate_heatmaps(np.array(l_2d_joint)[:,:,:2], heatmap_size=self.map_size, sigma=5)
            self.heatmap = self.heatmap_generator(np.array(l_2d_joint)[:,:,:2])
                    
        return thermal_map, ambient_temperature, l_depth_map, l_2d_joint, l_2d_flag, l_3d_joint, l_3d_flag, l_hand_depth, l_left_right_flag, self.heatmap


if __name__ == '__main__':
    # Test data loader
    path = 'Dataset/'
    file_names = os.listdir(path)
    # maps, ambient_temperatures, l_depth_maps, l_2d_joints, l_3d_joints, l_hand_depths, l_left_right_flag = data_loader(path, file_names[:3])
    # print(file_names[:3])
    # print(maps.shape)
    # print(ambient_temperatures.shape)
    # print(l_depth_maps.shape)
    # print(len(l_2d_joints))
    # print(len(l_3d_joints)) 
    # print(len(l_hand_depths))
    # print(l_left_right_flag)
    
    # print("test dataset class")
    # test dataset
    # transform = RandomOrthogonalRotation()
    dataset = iHand_dataset(path, file_names[:3],fragment_size = 5,map_size = (96,96), transform=None, require_heatmap=True, filter_out_no_hand=True, shuffle_fragment=True)
    print(len(dataset))
    thermal_map, ambient_temperature, l_depth_map, l_2d_joint, l_2d_flag, l_3d_joint, l_3d_flag, l_hand_depth, l_left_right_flag, heatmap = dataset[0]
    print(thermal_map.shape)
    print(ambient_temperature.shape)
    print(l_depth_map.shape)
    print(l_2d_joint.shape)
    print(l_2d_flag)
    print(l_3d_joint.shape)
    print(l_3d_flag)
    print(l_hand_depth)
    print(l_left_right_flag)
    print(heatmap.shape)
    