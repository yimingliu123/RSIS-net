import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
affine_par = True
from pretrainedmodels import inceptionresnetv2
import functools
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def concatenate(inputs,axis):
    h, w = 0, 0
    for i in inputs:
        if i.shape[2] > h:
            h = i.shape[2]
        if i.shape[3] > w:
            w = i.shape[3]
    upsample = []
    for i in inputs:
        upsample.append(nn.UpsamplingBilinear2d(size=(h, w))(i))
    return torch.cat(upsample,axis)


class ChannelwiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelwiseAttention, self).__init__()

        self.in_channels = in_channels  #C3:128+C4:128=C34:256

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels // 4)
        self.linear_2 = nn.Linear(self.in_channels // 4, self.in_channels)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))

        # Activity regularizer
        ca_act_reg = torch.mean(feats)

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()

        return feats, ca_act_reg

class backbone(nn.Module):

    def __init__(self, num_filters=256):

        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        self.inception = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(
            self.inception.conv2d_2a,
            self.inception.conv2d_2b,
            self.inception.maxpool_3a,
        )  # 64
        self.enc2 = nn.Sequential(
            self.inception.conv2d_3b,
            self.inception.conv2d_4a,
            self.inception.maxpool_5a,
        )  # 192
        self.enc3 = nn.Sequential(
            self.inception.mixed_5b,
            self.inception.repeat,
            self.inception.mixed_6a,
        )  # 1088
        self.enc4 = nn.Sequential(
            self.inception.repeat_1,
            self.inception.mixed_7a,
        )  # 2080

        self.pad = nn.ReflectionPad2d(1)
        self.lateral4 = nn.Conv2d(2080, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(32, num_filters // 2, kernel_size=1, bias=False)

        # # cfe2
        # cfe3
        self.conv21 = nn.Conv2d(1088, 32, (1, 1), padding=0)
        self.conv22 = nn.Conv2d(1088, 32, (3, 3), dilation=3, padding=3)
        self.conv23 = nn.Conv2d(1088, 32, (3, 3), dilation=5, padding=5)
        self.conv24 = nn.Conv2d(1088, 32, (3, 3), dilation=7, padding=7)
        self.bn5 = nn.BatchNorm2d(num_features=128, affine=False)
        # cfe4
        self.conv25 = nn.Conv2d(2080, 32, (1, 1), padding=0)
        self.conv26 = nn.Conv2d(2080, 32, (3, 3), dilation=3, padding=3)
        self.conv27 = nn.Conv2d(2080, 32, (3, 3), dilation=5, padding=5)
        self.conv28 = nn.Conv2d(2080, 32, (3, 3), dilation=7, padding=7)
        self.bn6 = nn.BatchNorm2d(num_features=128, affine=False)
        # channel wise attention
        self.cha_att = ChannelwiseAttention(in_channels=256)   #### 这里的 256 =C3:128+C4:128  128 = 32+32+32+32 这里是可以调整的
        # self.linear1 = nn.Linear(256, 56)
        # self.linear2 = nn.Linear(56, 256)
        self.conv29 = nn.Conv2d(256, 256, (1, 1), padding=0)
        self.bn7 = nn.BatchNorm2d(num_features=256, affine=False)
        self.relu = nn.ReLU()
        for param in self.inception.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)  #32
        # print("enc0 ")
        # print("enc0 ", enc0.size())
        enc1 = self.enc1(enc0)  # 64
        # print("enc1 ", enc1.size())
        enc2 = self.enc2(enc1)  # 192
        # print("enc2 ", enc2.size())
        enc3 = self.enc3(enc2)  # 1088
        # print("enc3 ", enc3.size())
        enc4 = self.enc4(enc3)  # 2080
        # print("enc4 ", enc4.size())


####################  CA

        C3_cfe = self.relu(
            self.bn5(concatenate([self.conv21(enc3), self.conv22(enc3), self.conv23(enc3), self.conv24(enc3)], 1)))

        C4_cfe = self.relu(
            self.bn6(concatenate([self.conv25(enc4), self.conv26(enc4), self.conv27(enc4), self.conv28(enc4)], 1)))
        #print(C3_cfe.size())
        #print(C4_cfe.size())
        C34 = concatenate([ C3_cfe, C4_cfe], 1)  # C34: [-1, 256*4*2 = 2048, h/4, w/4]
        #print(C34.size())
        conv_34_ca, ca_act_reg = self.cha_att(C34)
        conv_34_feats = torch.mul(conv_34_ca, ca_act_reg)

        ##################### 第一次自己写的CA
        # print(C45.size())
        # _h, _w = C45.shape[2:]
        # CA = nn.AvgPool2d(_h * _w)(C45)
        # print(CA.size())
        # CA = CA.view(-1, 256)
        # CA = self.linear1(CA)
        # CA = self.linear2(CA).view((-1, 256, 1, 1)).repeat([1, 1, _h, _w])
        # # # channel wise attention
        #
        # C45 = CA * C45
        ##################### 第一次自己写的CA

        C34 = self.conv29(conv_34_feats)  ##################这个卷积是有争议的  应该怎么选择????
        C34 = self.relu(self.bn7(C34))  # C345: [-1, 64, h/4, w/4]

        # print("C45 size ", C45.size())


####################  CA



        # Lateral connections

        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)


        lateral = []


        lateral.append(lateral4)
        lateral.append(lateral3)
        lateral.append(lateral2)
        lateral.append(lateral1)
        lateral.append(lateral0)


        infos = []
        for k in range(1,4):
            infos.append(F.interpolate(C34, lateral[k].size()[2:], mode='bilinear', align_corners=True))
        # print("######################################### ")
        # print("infos[0] ", infos[0].size())
        # print("infos[1] ", infos[1].size())
        # print("infos[2] ", infos[2].size())

        return  lateral , infos




def semantic():
    model = backbone()
    return model
