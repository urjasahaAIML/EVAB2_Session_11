"""
This module defines the DavidNet model
"""
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module): # Basic blocks
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        #print(f'Basic Block:1  {x.shape}')
        
        out = self.conv1(x)
        #print('Basic Block:2 ', x.shape)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module): # Resnet18 model
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 128  
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #print('gojng to call _make_layer1')
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        #self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        #print('gojng to call _make_layer3')
        self.in_planes = 512 
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=1)
        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        
        # custom for non-residual block
        
        # layer1:
        self.layer1_conv = nn.Conv2d(64, 128, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1_bn = nn.BatchNorm2d(128)
        
        # layer2:
        self.layer2_conv = nn.Conv2d(128, 256, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer2_bn = nn.BatchNorm2d(256)
        
        
        # layer3:
        self.layer3_conv = nn.Conv2d(256, 512, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer3_bn = nn.BatchNorm2d(512)
  
        
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        #print('strides ', strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            #print('self.in_planes ', self.in_planes)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # Prep Layer 64k
        #print('0', out.shape)
        
        # Add(X, layer1) 128k
        out = self.layer1_conv(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.layer1_bn(out))
        #print('1_0', out.shape)
        out = self.layer1(out)
        #print('1_1', out.shape)
        
        # layer2 256k
        out = self.layer2_conv(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.layer2_bn(out))
        #print('2', out.shape)
       
        # Add(X, layer3)512k
        out = self.layer3_conv(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.layer3_bn(out))
        #print('3_0', out.shape)
        
        out = self.layer3(out)
        #print('3_1', out.shape)
        
        
        #out = self.layer4(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def ResNet18(): # function used by other modules to create a model
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

