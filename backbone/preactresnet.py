'''
bn-->relu-->conv
'''

import torch.nn as nn

def Conv1(in_planes,planes,stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes,planes,7,stride=stride,padding=3,bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_planes,planes,stride=1,downsampling=False,expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes,planes,kernel_size=1,stride=1,bias=False),

            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,bias=False),

            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes*self.expansion, kernel_size=1, stride=1, bias=False),

        )

        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(planes),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
        #     nn.BatchNorm2d(planes),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(planes * self.expansion),
        #
        # )
        if self.downsampling:
            self.downsample = nn.Conv2d(in_planes,planes*self.expansion,kernel_size=1,stride=stride,bias=False)
        # if self.downsampling:
        #     self.downsample = nn.Sequential(
        #         nn.Conv2d(in_planes,planes*self.expansion,kernel_size=1,stride=stride,bias=False),
        #         nn.BatchNorm2d(planes*self.expansion)
        #     )
        # self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        if self.downsampling:
            residual = self.downsample(x)
        return out+residual
        # residual = x
        # out = self.bottleneck(x)
        # if self.downsampling:
        #     residual = self.downsample(x)
        # out += residual
        # out = self.relu(out)
        # return out

class PreactResnet(nn.Module):
    def __init__(self,blocks,num_class=10,expansion=4):
        super(PreactResnet, self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=3,planes=64)

        self.layer1 = self.make_layer(inplanes=64,planes=64,block=blocks[0],stride=1)
        self.layer2 = self.make_layer(inplanes=256, planes=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(inplanes=512, planes=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(inplanes=1024, planes=512, block=blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_class)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def make_layer(self,inplanes,planes,block,stride):
        layers = []
        layers.append(Bottleneck(in_planes=inplanes,planes=planes,stride=stride,downsampling=True))
        for i in range(1,block):
            layers.append(Bottleneck(planes*self.expansion,planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        print(x.size())
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

print(PreactResnet([3,4,6,3]))