from . import Conv2d
from . import BatchNorm2d
from . import Linear
from . import AvgPool2d
from . import MaxPool2d
from ..autodp.phi_tensor import PhiTensor

import torch


class Conv2d(torch.nn.Module):

    def __init__(self):
        super(Conv2d, self).__init__()




class ConvNet(torch.nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2)
        self.conv4 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2)
        self.conv5 = Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2)
        self.bn1 = BatchNorm2d(32)
        self.bn2 = BatchNorm2d(64)
        self.bn3 = BatchNorm2d(128)
        self.bn4 = BatchNorm2d(256)
        self.bn5 = BatchNorm2d(512)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.avg = AvgPool2d(7)
        self.fc = Linear(512 * 1 * 1, 2)


    def forward(self, image: PhiTensor):
        pass


