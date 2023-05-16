import torch
import torch.nn as nn
from torch.nn import functional as F


class CReLU(nn.Module):

    def __init__(self, inplace=False):
        super(CReLU, self).__init__()

    def forward(self, x):
        x = torch.cat((x, -x), 1)
        return F.relu(x)


class ClearView(nn.Module):
    def __init__(self):
        super(ClearView, self).__init__()

        self.relu = nn.Mish(inplace=True)
        self.depthwise1 = nn.Conv2d(
            3, 3, kernel_size=1, stride=1, padding=0, groups=1)
        self.point1 = nn.Conv2d(3, 3, kernel_size=1)

        self.depthwise2 = nn.Conv2d(
            3, 3, kernel_size=3, stride=1, padding=1, groups=1)
        self.point2 = nn.Conv2d(3, 3, kernel_size=1)

        self.depthwise3 = nn.Conv2d(
            6, 3, kernel_size=5, stride=1, padding=2, groups=1)
        self.point3 = nn.Conv2d(3, 3, kernel_size=1)

        self.depthwise4 = nn.Conv2d(
            6, 3, kernel_size=7, stride=1, padding=3, groups=1)
        self.point4 = nn.Conv2d(3, 3, kernel_size=1)

        self.depthwise5 = nn.Conv2d(
            12, 3, kernel_size=3, stride=1, padding=1, groups=1)
        self.point5 = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x):

        x1 = self.relu(self.point1(self.depthwise1(x)))
        x2 = self.relu(self.point2(self.depthwise2(x1)))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.point3(self.depthwise3(concat1)))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.point4(self.depthwise4(concat2)))

        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.point5(self.depthwise5(concat3)))

        clean_image = self.relu((x5 * x) - x5 + 1)

        return clean_image


class LightDehazeNet(nn.Module):

    def __init__(self):
        super(LightDehazeNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.e_conv_layer1 = nn.Conv2d(3, 8, 1, 1, 0, bias=True)
        self.e_conv_layer2 = nn.Conv2d(8, 8, 3, 1, 1, bias=True)
        self.e_conv_layer3 = nn.Conv2d(8, 8, 5, 1, 2, bias=True)
        self.e_conv_layer4 = nn.Conv2d(16, 16, 7, 1, 3, bias=True)
        self.e_conv_layer5 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.e_conv_layer6 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.e_conv_layer7 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv_layer8 = nn.Conv2d(56, 3, 3, 1, 1, bias=True)

    def forward(self, img):
        pipeline = []
        pipeline.append(img)

        conv_layer1 = self.relu(self.e_conv_layer1(img))
        conv_layer2 = self.relu(self.e_conv_layer2(conv_layer1))
        conv_layer3 = self.relu(self.e_conv_layer3(conv_layer2))
        concat_layer1 = torch.cat((conv_layer1, conv_layer3), 1)
        conv_layer4 = self.relu(self.e_conv_layer4(concat_layer1))
        conv_layer5 = self.relu(self.e_conv_layer5(conv_layer4))
        conv_layer6 = self.relu(self.e_conv_layer6(conv_layer5))
        concat_layer2 = torch.cat((conv_layer4, conv_layer6), 1)
        conv_layer7 = self.relu(self.e_conv_layer7(concat_layer2))
        concat_layer3 = torch.cat((conv_layer2, conv_layer5, conv_layer7), 1)
        conv_layer8 = self.relu(self.e_conv_layer8(concat_layer3))
        dehaze_image = self.relu((conv_layer8 * img) - conv_layer8 + 1)
        return dehaze_image
