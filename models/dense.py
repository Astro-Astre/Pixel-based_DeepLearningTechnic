import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from args import *


class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=True) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class DenseBlock(nn.Module):

    def __init__(self, input_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.k0 = input_channels
        self.k = growth_rate
        self.layers = self.__make_layers()

    def __make_layers(self):
        layer_list = []
        for i in range(self.num_layers):
            layer_list.append(nn.Sequential(
                BN_Conv2d(self.k0 + i * self.k, 4 * self.k, 1, 1, 0),
                BN_Conv2d(4 * self.k, self.k, 3, 1, 1)
            ))
        return layer_list

    def forward(self, x):
        feature = self.layers[0](x)
        out = torch.cat((x, feature), 1)
        for i in range(1, len(self.layers)):
            feature = self.layers[i](out)
            out = torch.cat((feature, out), 1)
        return out


class CSP_DenseBlock(nn.Module):

    def __init__(self, in_channels, num_layers, k, part_ratio=0.5):
        super(CSP_DenseBlock, self).__init__()
        self.part1_chnls = int(in_channels * part_ratio)
        self.part2_chnls = in_channels - self.part1_chnls
        self.dense = DenseBlock(self.part2_chnls, num_layers, k)
        # trans_chnls = self.part2_chnls + k * num_layers
        # self.transtion = BN_Conv2d(trans_chnls, trans_chnls, 1, 1, 0)

    def forward(self, x):
        part1 = x[:, :self.part1_chnls, :, :]
        part2 = x[:, self.part1_chnls:, :, :]
        part2 = self.dense(part2)
        # part2 = self.transtion(part2)
        out = torch.cat((part1, part2), 1)
        return out


class DenseNet(nn.Module):

    def __init__(self, layers: object, k, theta, num_classes, part_ratio=0) -> object:
        super(DenseNet, self).__init__()
        # params
        self.layers = layers
        self.k = k
        self.theta = theta
        self.Block = DenseBlock if part_ratio == 0 else CSP_DenseBlock  # 通过part_tatio参数控制block type
        # layers
        self.conv = BN_Conv2d(3, 2 * k, 7, 2, 3)
        self.blocks, patches = self.__make_blocks(2 * k)
        self.fc = nn.Linear(patches, num_classes)

    def __make_transition(self, in_chls):
        out_chls = int(self.theta * in_chls)
        return nn.Sequential(
            BN_Conv2d(in_chls, out_chls, 1, 1, 0),
            nn.AvgPool2d(2)
        ), out_chls

    def __make_blocks(self, k0):
        """
        make blocks-transition structures
        :param k0:
        :return:
        """
        layers_list = []
        patches = 0
        for i in range(len(self.layers)):
            layers_list.append(self.Block(k0, self.layers[i], self.k))
            patches = k0 + self.layers[i] * self.k  # output feature patches from Dense Block
            if i != len(self.layers) - 1:
                transition, k0 = self.__make_transition(patches)
                layers_list.append(transition)
        return nn.Sequential(*layers_list), patches

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(out, 3, 2, 1)
        # print(out.shape)
        out = self.blocks(out)
        # print(out.shape)
        out = F.avg_pool2d(out, 7)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


num_classes = data_config.num_class


def densenet_121():
    return DenseNet([6, 12, 24, 16], k=32, theta=0.5, num_classes=num_classes)


def densenet_169():
    return DenseNet([6, 12, 32, 32], k=32, theta=0.5, num_classes=num_classes)


def densenet_201():
    return DenseNet([6, 12, 48, 32], k=32, theta=0.5, num_classes=num_classes)


def densenet_264():
    return DenseNet([6, 12, 64, 48], k=32, theta=0.5, num_classes=num_classes)


def csp_densenet_121():
    return DenseNet([6, 12, 24, 16], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)


def csp_densenet_169():
    return DenseNet([6, 12, 32, 32], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)


def csp_densenet_201():
    return DenseNet([6, 12, 48, 32], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)


def csp_densenet_264():
    return DenseNet([6, 12, 64, 48], k=32, theta=0.5, num_classes=num_classes, part_ratio=0.5)
