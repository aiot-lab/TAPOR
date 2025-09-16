# this is the implementation of the ColorHandPose3D model: 
# Zimmermann, Christian, and Thomas Brox. "Learning to estimate 3d hand pose from single rgb images." Proceedings of the IEEE international conference on computer vision. 2017.
# we modified this model to fit our dataset


import os
import torch
# from .HandSegNet import HandSegNet
from .PoseNet import PoseNet
from .PosePrior import PosePrior
# from .ViewPoint import ViewPoint
from .utils.general import *
from .utils.transforms import *


class ColorHandPose3D(torch.nn.Module):
    """ColorHandPose3D predicts the 3D joint location of a hand given the
    cropped color image of a hand."""

    def __init__(self,in_channels=1, weight_path=None, crop_size=None, num_keypoints=None):
        super(ColorHandPose3D, self).__init__()
        # self.handsegnet = HandSegNet()
        self.posenet = PoseNet(in_channels=in_channels)
        self.poseprior = PosePrior()
        # self.viewpoint = ViewPoint()

        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size

        if num_keypoints is None:
            self.num_keypoints = 21
        else:
            self.num_keypoints = num_keypoints

        # Load weights
        if weight_path is not None:
            self.handsegnet.load_state_dict(
                    torch.load(os.path.join(weight_path, 'handsegnet.pth.tar')))
            self.posenet.load_state_dict(
                    torch.load(os.path.join(weight_path, 'posenet.pth.tar')))
            self.poseprior.load_state_dict(
                    torch.load(os.path.join(weight_path, 'poseprior.pth.tar')))
            self.viewpoint.load_state_dict(
                    torch.load(os.path.join(weight_path, 'viewpoint.pth.tar')))

    def forward(self, x, hand_sides):
        """Forward pass through the network.

        Args:
            x - Tensor (B x C x H x W): Batch of images.
            hand_sides - Tensor (B x 2): One-hot vector indicating if the hand
                is left or right.

        Returns:
            coords_xyz_rel_normed (B x N_k x 3): Normalized 3D coordinates of
                the joints, where N_k is the number of keypoints.
        """

        # Segment the hand
        # hand_scoremap = self.handsegnet.forward(x)

        # # Calculate single highest scoring object
        # hand_mask = single_obj_scoremap(hand_scoremap, self.num_keypoints)

        # # crop and resize
        # centers, _, crops = calc_center_bb(hand_mask)
        # crops = crops.to(torch.float32)

        # crops *= 1.25
        # scale_crop = torch.min(
        #         torch.max(self.crop_size / crops,
        #             torch.tensor(0.25, device=x.device)),
        #         torch.tensor(5.0, device=x.device))
        # image_crop = crop_image_from_xy(x, centers, self.crop_size, scale_crop)

        # detect 2d keypoints
        keypoints_scoremap = self.posenet(x)

        # estimate 3d pose
        coord_can = self.poseprior(keypoints_scoremap, hand_sides)

        # rot_params = self.viewpoint(keypoints_scoremap, hand_sides)
        # get normalized 3d coordinates
        # rot_matrix = get_rotation_matrix(rot_params)
        # cond_right = torch.eq(torch.argmax(hand_sides, 1), 1)
        # cond_right_all = torch.reshape(cond_right, [-1, 1, 1]).repeat(1, self.num_keypoints, 3)
        # coords_xyz_can_flip = flip_right_hand(coord_can, cond_right_all)
        # coords_xyz_rel_normed = coords_xyz_can_flip @ rot_matrix

        # flip left handed inputs wrt to the x-axis for Libhand compatibility.
        # coords_xyz_rel_normed = flip_left_hand(coords_xyz_can_flip, cond_right_all)

        # scale heatmaps
        keypoints_scoremap = F.interpolate(keypoints_scoremap,
                                           self.crop_size,
                                           mode='bilinear',
                                           align_corners=False)
        # print(coords_xyz_rel_normed.shape, keypoints_scoremap.shape)
        return coord_can, keypoints_scoremap.to(torch.float32) 
        # correspond to the 3d coordinates of the 21 joints of the hand and the 2d coordinates heatmap of the 21 joints of the hand
    


if __name__ == '__main__':
    # Test the network
    # from torchinfo import summary
    net = ColorHandPose3D(in_channels=1)
    coord_can, keypoints_scoremap = net(torch.randn(2, 1, 256, 256), torch.randn(2, 2))
    print(coord_can.shape, keypoints_scoremap.shape)
    # torch.Size([2, 21, 3]) torch.Size([2, 21, 256, 256])
    # summary(net, (1,3, 256, 256))
