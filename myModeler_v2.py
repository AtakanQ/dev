import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

        self.conv2 = nn.Conv2d(3,32, 3, stride = 1,padding=1)#32x32x32
        self.conv3 = nn.Conv2d(3,32, 5, stride = 1,padding=2)#32x32x32
        self.conv4 = nn.Conv2d(3,32, 3, stride = 1,padding=1,dilation=1)#32x32x32
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.batchNorm3 = nn.BatchNorm2d(32)
        self.batchNorm4 = nn.BatchNorm2d(32)

        #MAXPOOL

        self.conv22 = nn.Conv2d(32,32, 3, stride = 1,padding=1)#16x16x32
        self.conv33 = nn.Conv2d(32,32, 3, stride = 1,padding=1)#16x16x32
        self.conv44 = nn.Conv2d(32,32, 3, stride = 1,padding=1)#16x16x32
        self.batchNorm22 = nn.BatchNorm2d(32)
        self.batchNorm33 = nn.BatchNorm2d(32)
        self.batchNorm44 = nn.BatchNorm2d(32)


        self.conv222 = nn.Conv2d(32,64, 3, stride = 1,padding=1)#16x16x64
        self.conv333 = nn.Conv2d(32,64, 3, stride = 1,padding=1)#16x16x64
        self.conv444 = nn.Conv2d(32,64, 3, stride = 1,padding=1)#16x16x64
        self.batchNorm222 = nn.BatchNorm2d(64)
        self.batchNorm333 = nn.BatchNorm2d(64)
        self.batchNorm444 = nn.BatchNorm2d(64)

        #MAXPOOL HERE
        self.conv2222 = nn.Conv2d(64,128, 3, stride = 2,padding=1)#8x8x128
        self.conv3333 = nn.Conv2d(64,128, 3, stride = 2,padding=1)#8x8x128
        self.conv4444 = nn.Conv2d(64,128, 3, stride = 2,padding=1)#8x8x128
        self.batchNorm2222 = nn.BatchNorm2d(128)
        self.batchNorm3333 = nn.BatchNorm2d(128)
        self.batchNorm4444 = nn.BatchNorm2d(128)


        #GAP
        #128*3 = 384

        self.fc1 = nn.Linear(128*3,128)
        self.batchNormfc1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128,32)
        self.batchNormfc2 = nn.BatchNorm1d(32)

        self.fc3 = nn.Linear(32,10)

        self.maxPooler = nn.MaxPool2d(2,2)
        self.avgPooler = nn.AvgPool2d(2,2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.leakyRelu = nn.LeakyReLU(0.01) # leaky?

        self.dropOut = nn.Dropout(0.3)

    def forward(self, x):
        y2_1 = self.conv2(x) #32x32x32
        y3_1 = self.conv3(x) #32x32x32
        y4_1 = self.conv4(x) #32x32x32

        y2_1 = self.leakyRelu(self.batchNorm2(y2_1))
        y3_1 = self.leakyRelu(self.batchNorm3(y3_1))
        y4_1 = self.leakyRelu(self.batchNorm4(y4_1))

        y2_1 = self.maxPooler(y2_1) #16x16x32
        y3_1 = self.maxPooler(y3_1) #16x16x32
        y4_1 = self.maxPooler(y4_1) #16x16x32

        y2_1 = self.leakyRelu(self.batchNorm22(self.conv22(y2_1))) #16x16x32
        y3_1 = self.leakyRelu(self.batchNorm33(self.conv33(y3_1))) #16x16x32
        y4_1 = self.leakyRelu(self.batchNorm44(self.conv44(y4_1))) #16x16x32

        y2_1 = self.maxPooler(y2_1) #8x8x64
        y3_1 = self.maxPooler(y3_1) #8x8x64
        y4_1 = self.maxPooler(y4_1) #8x8x64

        y2_1 = self.leakyRelu(self.batchNorm222(self.conv222(y2_1))) #8x8x64
        y3_1 = self.leakyRelu(self.batchNorm333(self.conv333(y3_1))) #8x8x64
        y4_1 = self.leakyRelu(self.batchNorm444(self.conv444(y4_1))) #8x8x64

        y2_1 = self.maxPooler(y2_1) #4x4x64
        y3_1 = self.maxPooler(y3_1) #4x4x64
        y4_1 = self.maxPooler(y4_1) #4x4x64

        y2_1 = self.leakyRelu(self.batchNorm2222(self.conv2222(y2_1))) #4x4x128
        y3_1 = self.leakyRelu(self.batchNorm3333(self.conv3333(y3_1))) #4x4x128
        y4_1 = self.leakyRelu(self.batchNorm4444(self.conv4444(y4_1))) #4x4x128
        
        y2_1gap = self.gap(y2_1) #1x1x128
        y3_1gap = self.gap(y3_1) #1x1x128
        y4_1gap = self.gap(y4_1) #1x1x128
        
        yx_6_all = torch.cat((y2_1gap,y3_1gap,y4_1gap),dim=1) #1x1x384
        yx_6_all = torch.flatten(yx_6_all, start_dim=1) #384

        fc1 = self.fc1(yx_6_all)
        fc1 = self.batchNormfc1(fc1)
        fc1 = self.leakyRelu(fc1)
        fc1 = self.dropOut(fc1)


        fc2 = self.fc2(fc1)
        fc2 = self.batchNormfc2(fc2)
        fc2 = self.leakyRelu(fc2)
        fc2 = self.dropOut(fc2)


        y = self.fc3(fc2)


        return y

