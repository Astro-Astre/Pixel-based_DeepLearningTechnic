# -*- coding: utf-8-*-
from torch import nn


class DecalsModel(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self):
        super(DecalsModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 6, stride=1, padding=3, bias=True),
            nn.MaxPool2d(20, stride=2, padding=9),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=True),
            nn.MaxPool2d(8, stride=2, padding=4),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, bias=True),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(131072, 128, bias=True),
            nn.Dropout(0.5),
            nn.Linear(128, 128, bias=True),
            nn.Linear(128, 12, bias=True),
        )

    def forward(self, x):
        batch_size = x.size(0)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        # print('第四层：',out.shape)
        out = out_conv4.view(batch_size, -1)
        out = self.fc(out)
        return out
