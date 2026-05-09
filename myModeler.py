import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

        self.conv1 = nn.Conv2d(3,4, 1, stride = 1,padding=0)
        self.conv2 = nn.Conv2d(3,16, 3, stride = 1,padding=1)
        self.conv3 = nn.Conv2d(3,4, 5, stride = 1,padding=2)

        self.batchNorm1 = nn.BatchNorm2d(4)
        self.batchNorm2 = nn.BatchNorm2d(16)
        self.batchNorm3 = nn.BatchNorm2d(4)


        self.convChannelRel = nn.Conv2d(48,12,1,1) # Changed in_channels from 24 to 48
        self.batchNormRelatives = nn.BatchNorm2d(12)

        self.fc1 = nn.Linear(12*8*8,256)
        self.batchNormfc1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256,32)
        self.batchNormfc2 = nn.BatchNorm1d(32)

        self.fc3 = nn.Linear(32,10)
        self.batchNormfc3 = nn.BatchNorm1d(10)

        self.maxPooler = nn.MaxPool2d(2,2)
        self.avgPooler = nn.AvgPool2d(2,2)

        self.leakyRelu = nn.LeakyReLU(0.01) # leaky?
        self.softMax = nn.Softmax()

        self.dropOut = nn.Dropout(0.3)

    def forward(self, x):
        y1_1 = self.conv1(x) #32x32x4
        y2_1 = self.conv2(x) #32x32x16
        y3_1 = self.conv3(x) #32x32x4

        y1_1 = self.leakyRelu(self.batchNorm1(y1_1))
        y2_1 = self.leakyRelu(self.batchNorm2(y2_1))
        y3_1 = self.leakyRelu(self.batchNorm3(y3_1))

        y1_2max = self.maxPooler(y1_1) #16x16x4
        y2_2max = self.maxPooler(y2_1) #16x16x16
        y3_2max = self.maxPooler(y3_1) #16x16x4

        y1_2avg = self.avgPooler(y1_1) #16x16x4
        y2_2avg = self.avgPooler(y2_1) #16x16x16
        y3_2avg = self.avgPooler(y3_1) #16x16x4

        yx_3_all = torch.cat([y1_2max, y2_2max, y3_2max, y1_2avg , y2_2avg, y3_2avg],dim = 1 ) # 16x16x24


        yx_4_all = self.convChannelRel(yx_3_all) #take the channel relations 16x16x12
        yx_4_all = self.batchNormRelatives(yx_4_all)
        yx_4_all = self.leakyRelu(yx_4_all)


        yx_4_all = self.maxPooler(yx_4_all) # 8x8x12
        yx_4_all = torch.flatten(yx_4_all, start_dim=1) #768

        fc1 = self.fc1(yx_4_all)
        fc1 = self.batchNormfc1(fc1)
        fc1 = self.leakyRelu(fc1)
        fc1 = self.dropOut(fc1)


        fc2 = self.fc2(fc1)
        fc2 = self.batchNormfc2(fc2)
        fc2 = self.leakyRelu(fc2)
        fc2 = self.dropOut(fc2)


        y = self.fc3(fc2)
        y = self.batchNormfc3(y)
        y = self.leakyRelu(y)


        return y

