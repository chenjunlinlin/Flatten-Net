import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from .non_local import NONLocalBlock2D


class ResNet_NonLocal(nn.Module):

    def __init__(self,resnet_model):
        super(ResNet_NonLocal, self).__init__()

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = list(resnet_model.children())[2]

        self.nonlocal_layer = NONLocalBlock2D(in_channels=512, sub_sample=False)
        self.maxpool = list(resnet_model.children())[3]
        self.layer1 = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2 = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3 = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4 = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = list(resnet_model.children())[8]
        self.fc = list(resnet_model.children())[9]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.nonlocal_layer(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x