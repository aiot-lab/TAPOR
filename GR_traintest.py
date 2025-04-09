import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import random
import pickle   
from tqdm import tqdm
import torch.nn as nn
import torch
import argparse
import time
from tapor_model_config import small_setting, base1_setting, base2_setting, large1_setting, large2_setting, base1_setting_varaint1, base1_setting_varaint2
from models import BlazeHandLandmark, ColorHandPose3D, Mano_MobileNetV2, Tapor
from torch.utils.tensorboard import SummaryWriter

######################### create the dataset class for the gesture recognition task #########################
def data_loader(path, file_names):
    # file_name to gesture label mapping
    filename_to_label = {
        'gesture0': 0,
        'gesture1': 1,
        'gesture2': 2,
        'gesture3': 3,
        'gesture4': 4,
        'gesture5': 5,
        'gesture6': 6,
    }
    maps = []
    ambient_temperatures = []
    skip_the_start_frames = 20
    labels = []
    for file_name in file_names:
        with open(os.path.join(path, file_name), 'rb') as f:
            data = pickle.load(f)
        gesture_label = None
        for key in filename_to_label.keys():
            if key in file_name:
                gesture_label = [filename_to_label[key] for _ in range(len(data['thermal_array_temperature_maps'][skip_the_start_frames:]))]
                break
        if gesture_label is None:
            print("No label found for the file: ", file_name)
        else:
            labels.append(np.array(gesture_label))
            maps.append(np.array(data['thermal_array_temperature_maps'][skip_the_start_frames:]))
            ambient_temperatures.append(np.array(data['thermal_array_ambient_temperatures'][skip_the_start_frames:]))
    return np.concatenate(maps), np.concatenate(ambient_temperatures), np.concatenate(labels)


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


class Gesture_dataset(Dataset):
    def __init__(self, path, file_names, fragment_size = 1,map_size = (96,96), shuffle_fragment=False,sliding_window=False):
        super(Gesture_dataset, self).__init__()
        """
        params:
            path: the path of the dataset
            file_names: the list of file names that used to construct the dataset from the path folder
            fragment_size: the number of thermal array maps in one sample, can be regarded as the channels of the input
            map_size: the size of the interpolated thermal map
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
        self.maps, self.ambient_temperatures, self.labels = data_loader(path, file_names)
        self.shuffle_fragment = shuffle_fragment   
        # preprocessing the data
        self.maps = np.array([cv2.resize(SubpageInterpolating(thermal_map), self.map_size, interpolation=cv2.INTER_NEAREST) for thermal_map in self.maps])
        # convert numpy.uint16 to numpy.float32
        self.labels = self.labels.astype(np.float32)
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
        labels = self.labels[start_index:end_index]
        return thermal_map, ambient_temperature, labels



###################### create the gesture classifer with nn module ######################
class GestureClassifier(nn.Module):
    def __init__(self, map_size=(32,24),  expanding_vector=2, num_classes=7, plug_thermal_map = False):
        super(GestureClassifier, self).__init__()
        self.num_classes = num_classes
        self.plug_thermal_map = plug_thermal_map

        if plug_thermal_map:
            self.map2hidden = nn.Sequential(
                nn.Flatten(),
                nn.Linear(map_size[0]*map_size[1], int(map_size[0]*map_size[1]*expanding_vector)),
                nn.ReLU(),
            )
            self.pose2hidden = nn.Sequential(
                nn.Flatten(),
                nn.Linear(21*3, int(21*3*expanding_vector)),
                nn.ReLU(),
            )
            self.hiddens2gesture = nn.Sequential(
                nn.Linear((int(map_size[0]*map_size[1]*expanding_vector) + int(21*3*expanding_vector)), num_classes)
            )
        else:
            self.pose2gesture = nn.Sequential(
            nn.Flatten(),
            nn.Linear(21*3, int(21*3*expanding_vector)),
            nn.ReLU(),
            nn.Linear( int(21*3*expanding_vector), num_classes)
        )  
    
    def forward(self, pose, thermal_map=None):
        if self.plug_thermal_map:
            thermal_map = self.map2hidden(thermal_map)
            pose = self.pose2hidden(pose)
            x = torch.cat((thermal_map, pose), dim=1)
            return self.hiddens2gesture(x)
        else:
            return self.pose2gesture(pose)
        


class EdgeTapor(nn.Module):
    def __init__(self, feat_dim =21*16, last_conv_channel=32):
        super(EdgeTapor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, last_conv_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(last_conv_channel*3*4, feat_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, 21*3)
        )
    
    def forward(self, thermal_map):
        feat = self.encoder(thermal_map)
        pose = self.decoder(feat)
        return pose


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

if __name__ == "__main__":
    # Note: EdgeTapor is the same as NanoTapor (these terms are used interchangeably)
    parser = argparse.ArgumentParser(description='Gesture Recognition: Training and Testing')
    parser.add_argument('-wp', '--model_weight_file_name', type=str, default='.pth', help='the file name of the weights of the Tapor model')
    parser.add_argument('-fs', '--fragment_size', type=int, default=1, help='the fragment size of the dataset, which is equal to the sequence length and the in_channel of the input')
    parser.add_argument('-sw', '--sliding_window', type=int, default=0, help='the sliding window size of the dataset')
    parser.add_argument('-ms', '--map_size', type=int, default=0, help='the size of the thermal array map (input)')
    parser.add_argument('-mt', '--tapor_type', type=int, default=1, help='the type of tapor model: 0: small, 1: base1, 2: base2,\
        3: large1, 4: large2,5: base1_varaint1, 6: base1_variant2; 8: edgeTapor')
    
    parser.add_argument('-ev', '--expanding_vector', type=int, default=1, help='the expanding vector of the hidden layers')
    parser.add_argument('-nc', '--num_classes', type=int, default=7, help='the number of classes of the gesture recognition task')
    parser.add_argument('-ptm', '--plug_thermal_map', type=int, default=0, help='whether to plug the thermal map into the model')
    
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='the learning rate of the optimizer')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='the batch size of the training and testing')
    parser.add_argument('-ep', '--epochs', type=int, default=100, help='the epochs of the training')
    parser.add_argument('-sd', '--seed', type=int, default=0, help='the random seed')
    parser.add_argument('-dd', '--device', type=int, default=0, help='the device id')
    
    parser.add_argument('-wpg', '--weight_path_gesture', default='', help='the path of the weights for resuming the training')
    
    args = parser.parse_args()
    
    model_weight_file_name = args.model_weight_file_name
    model_weight_path = "weights/" + model_weight_file_name # the path of the Tapor model weights
    fragment_size = args.fragment_size
    sliding_window = args.sliding_window
    map_size = args.map_size
    tapor_type = args.tapor_type
    
    expanding_vector = args.expanding_vector
    num_classes = args.num_classes
    plug_thermal_map = args.plug_thermal_map
    
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    seed = args.seed
    device = args.device
    
    weight_path_gesture = args.weight_path_gesture
    
    if tapor_type >= 8: # we use edgeTapor
        model_weight_file_name = args.model_weight_file_name
        model_weight_path = "NanoTapor_files/NanoTapor_model_weights/" + model_weight_file_name # the path of the EdgeTapor model weights
        print("The model weight path is: ",model_weight_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_ids(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The device is: ",device)
    # fix the seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # create folders to save the model weights and the training logs
    tensorboard_folder = 'gesture_runs'
    if not os.path.exists("gesture_model_weights"):
        os.makedirs("gesture_model_weights")
    if not os.path.exists("gesture_logs"):
        os.makedirs("gesture_logs")
    if not os.path.exists(tensorboard_folder):
        os.makedirs(tensorboard_folder)
        
    localtime = time.localtime(time.time())
    index_of_experiment = len(os.listdir(tensorboard_folder))
    # the name for both the log file and the tensorboard log
    if tapor_type >= 8: # now we using edgeTapor
        log_file_name = str(index_of_experiment) + "_edgeTapor_" + str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday) + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)
    else:
        log_file_name = str(index_of_experiment) + "_" + str(localtime.tm_year) + str(localtime.tm_mon) + str(localtime.tm_mday) + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)
    print("The log file name is: ",log_file_name)
    
    ## Loading the Tapor Model ##
    if tapor_type < 8:
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

        tapor_model = Tapor(spatial_encoder_param, 
                    keypoints_encoder_param, 
                    cross_keypoints_fusion_param, 
                    temporal_keypoints_fusion_param,
                    handpose_encoder_param,
                    input_width=32, 
                    input_height=24,
                    batch_size = batch_size,
                    train=True,
                    device=device).to(device)
        tapor_model.load_state_dict(torch.load(model_weight_path, map_location=device))
    else:
        tapor_model = EdgeTapor().to(device)
        tapor_model.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    
    ## create the classifier model ##
    classifier = GestureClassifier(map_size=(32,24),  
                                   expanding_vector=expanding_vector,
                                   num_classes=num_classes, 
                                   plug_thermal_map = plug_thermal_map
                                   ).to(device)
    
    if weight_path_gesture != '':
        classifier.load_state_dict(torch.load(weight_path_gesture, map_location=device))
    
    ## create the dataset ##
    data_path = "GR_Dataset/"
    file_names = os.listdir(data_path)
    dataset = Gesture_dataset(data_path, file_names, 
                              fragment_size, 
                              map_size=(32,24), 
                              shuffle_fragment=False, 
                              sliding_window=sliding_window
                              )
    print("The length of the dataset is: ",len(dataset))
    # split the dataset into training, validation and testing
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                                             [train_size, val_size, test_size]
                                                                             )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    ## optimizer and loss function ##
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print("start training...")
    ## training ##
    writer = SummaryWriter(tensorboard_folder + '/' + log_file_name)
    best_val_acc = 0
    best_epoch = 0
    best_weights = None
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            thermal_map, ambient_temperature, labels = data
            thermal_map = thermal_map.to(device)
            ambient_temperature = ambient_temperature.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                if tapor_type < 8:
                    pose, current_kp_feat, sp_feat, kp_feat, attention_map = tapor_model(thermal_map)
                else:
                    pose = tapor_model(thermal_map)
            optimizer.zero_grad()
            outputs = classifier(pose, thermal_map)
            loss = criterion(outputs, labels.long().squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                writer.add_scalar('training loss', running_loss / 10, epoch * len(train_loader) + i)
                running_loss = 0.0
        ## validation ##
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                thermal_map, ambient_temperature, labels = data
                thermal_map = thermal_map.to(device)
                ambient_temperature = ambient_temperature.to(device)
                labels = labels.to(device)
                if tapor_type < 8:
                    pose, current_kp_feat, sp_feat, kp_feat, attention_map = tapor_model(thermal_map)
                else:
                    pose = tapor_model(thermal_map)
                outputs = classifier(pose, thermal_map)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.long().squeeze()).sum().item()
        print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))
        if best_weights is not None:
            best_weights = classifier.state_dict()
        if best_val_acc < 100 * correct / total:
            best_val_acc = 100 * correct / total
            best_epoch = epoch
            best_weights = classifier.state_dict()
        writer.add_scalar('validation accuracy', 100 * correct / total, epoch)
    print('The best validation accuracy is: %d %% at epoch %d' % (best_val_acc, best_epoch))

    ## testing
    classifier.eval()
    correct = 0
    total = 0
    test_labels = []
    test_predicted = []
    # using the best weights
    classifier.load_state_dict(best_weights)
    with torch.no_grad():
        for data in test_loader:
            thermal_map, ambient_temperature, labels = data
            thermal_map = thermal_map.to(device)
            ambient_temperature = ambient_temperature.to(device)
            labels = labels.to(device)
            if tapor_type < 8:
                pose, current_kp_feat, sp_feat, kp_feat, attention_map = tapor_model(thermal_map)
            else:
                pose = tapor_model(thermal_map)
            outputs = classifier(pose, thermal_map)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long().squeeze()).sum().item()
            # get the labels and the predicted labels for later to save 
            test_labels.append(labels.long().squeeze().cpu().numpy())
            test_predicted.append(predicted.cpu().numpy())
    print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
    writer.add_scalar('test accuracy', 100 * correct / total, epoch)
    writer.close()
    
    # save the model weights
    torch.save(best_weights, "gesture_model_weights/" + log_file_name + ".pth")
    ## save the labels and the predicted labels in gesture_logs
    test_labels = np.concatenate(test_labels)
    test_predicted = np.concatenate(test_predicted)
    out_puts = {
        "test_labels": test_labels,
        "test_predicted": test_predicted
    }
    with open("gesture_logs/" + log_file_name + ".pkl", 'wb') as f:
        pickle.dump(out_puts, f)
    
    print("The training and testing is finished!")