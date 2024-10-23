"""
Please see the __main__() for sample operations to extract the feature map using a pretrained VGG-19 and ResNet-50 network.
"""


import numpy as np
import torch
import torch.nn as nn
from skimage import io, transform

import importlib

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # encode 1-1
            nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 1-1
            # encode 2-1
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), #1/2

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 2-1
            # encoder 3-1
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), #1/4
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 3-1
            # encoder 4-1
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), #1/8

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 4-1
            # rest of vgg not used
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), #1/16

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 5-1
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True)
        )

    def load_weights(self, path_to_weights):
        vgg_model = torch.load(path_to_weights)
        # Don't care about the extra weights
        self.model.load_state_dict(vgg_model, strict=False)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        # Input is numpy array of shape (H, W, 3)
        # Output is numpy array of shape (N_l, H_l, W_l)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        out = self.model(x)
        out = out.squeeze(0).numpy()
        return out
        

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)

class CustomResNet(nn.Module):
    def __init__(self,
                 encoder='resnet50',
                 pretrained=True):

        super(CustomResNet, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        # if encoder in ['resnet18', 'resnet34']:
        #     filters = [64, 128, 256, 512]
        # else:
        #     filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)

        for parameter in resnet.parameters():
            parameter.requires_grad = False

        self.firstconv = resnet.conv1  # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # H/4

        # encoder
        self.layer1 = resnet.layer1  # H/4
        self.layer2 = resnet.layer2  # H/8
        self.layer3 = resnet.layer3  # H/16

    def forward(self, x):
        """
        Coarse and Fine Feature extraction using ResNet
        Coarse Feature Map has smaller spatial sizes.
        Arg:
            x: (np.array) [H,W,C]
        Rerurn:
            xc: (np.array) [C_coarse, H/16, W/16]
            xf: (np.array) [C_fine, H/8, W/8]
        """
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()

        x = self.firstrelu(self.firstbn(self.firstconv(x))) #1/2
        x = self.firstmaxpool(x) #1/4

        x = self.layer1(x) #1/4
        xf = self.layer2(x) #1/8
        xc = self.layer3(xf) #1/16

        # convert xc, xf to numpy
        xc = xc.squeeze(0).numpy()
        xf = xf.squeeze(0).numpy()
        return xc, xf


if __name__ == '__main__':
    # ---------- Pre processing -----------------------#
    # Read image 
    x = io.imread('data/training/cloudy1.jpg')
    # Resize the input image
    x = transform.resize(x, (256, 256))    
    # ----------------- VGG19 Example -----------------#
    # Load the model and the provided pretrained weights
    vgg = VGG19()
    vgg.load_weights('vgg_normalized.pth')    
    # Obtain the output feature map
    vgg_feature = vgg(x)
    print(vgg_feature.shape)

    # ----------------- CustomResNet Example -----------------#
    encoder_name='resnet50' # Valid options ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    # CustomResNet will download the model weights from pytorch to the following path:
        # resnet50 ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth  Size - 98MB
    resnet = CustomResNet(encoder=encoder_name)
    resnet_feat_coarse, resnet_feat_fine = resnet(x)     
    print("Coarse Shape: ", resnet_feat_coarse.shape)
    print("Fine Shape: ", resnet_feat_fine.shape)

