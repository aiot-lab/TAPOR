import torch
import torch.nn as nn
import torchvision.models as models
import mano
from mano.utils import Mesh

class Mano_MobileNetV2(nn.Module):
    def __init__(self, num_classes=51,batch_size=10):
        super(Mano_MobileNetV2, self).__init__()
        self.model_path = "/home/shared/IRAHand/iHand/models/mano_base/mano/MANO_RIGHT.pkl"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rh_model = mano.load(model_path=self.model_path,
                                  is_right= True,
                                  num_pca_comps=45,
                                  batch_size=batch_size,
                                  flat_hand_mean=False)
        self.model_path = "/home/shared/IRAHand/iHand/models/mano_base/mano/MANO_LEFT.pkl"
        self.lh_model = mano.load(model_path=self.model_path,
                                  is_right= False,
                                  num_pca_comps=45,
                                  batch_size=batch_size,
                                  flat_hand_mean=False)
        # Load a pre-trained mobilenet_v2 model
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=False)

        # Modify the first convolution layer to accept 1 channel input
        self.mobilenet_v2.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Replace the classifier layer with a new one (output 61 classes instead of 1000)
        num_features = self.mobilenet_v2.classifier[1].in_features
        self.mobilenet_v2.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        self.betas = torch.ones(batch_size, 10, dtype=torch.float32).to(self.device)
        self.transl = torch.zeros(batch_size, 3, dtype=torch.float32).to(self.device)

    def forward(self, x):
        x=self.mobilenet_v2(x)
        output_l = self.lh_model(betas=self.betas,
                      global_orient=x[:,:3],
                      hand_pose=x[:,3:48],
                      transl=x[:,48:51],
                      return_verts=True,
                      return_tips = True)
        output_r = self.rh_model(betas=self.betas,
                      global_orient=x[:,:3],
                      hand_pose=x[:,3:48],
                      transl=x[:,48:51],
                      return_verts=True,
                      return_tips = True)
        
        return output_l.joints,output_r.joints

if __name__ == '__main__':
    net = Mano_MobileNetV2()
    joints_l, joints_r = net(torch.rand(10,1,96,72))
    print(joints_l.shape)
    print(joints_r.shape)