"""
This is the data loader module.
It uses albumentation augmentation.
"""

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import hyperparameters
from hyperparameters import * 

import albumentations as A
from albumentations.pytorch import ToTensor
import albumentations.augmentations.transforms as albtransforms 

    

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # normalize image
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

pmda_train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(),  
                         transforms.CenterCrop(32),                        
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats)])
train_transform = A.Compose([ 
    albtransforms.PadIfNeeded(min_height=40, min_width=40, border_mode=4, value=[0, 0, 0], always_apply=False, p=1.),                             
    #albtransforms.RandomCrop(32,32,always_apply=False, p=1.0),
    albtransforms.RandomCrop(32,32,always_apply=False, p=1.),
    #albtransforms.HorizontalFlip(1.0), 
    albtransforms.HorizontalFlip(0.5),    
    albtransforms.Cutout (num_holes=8, max_h_size=8, max_w_size=8, always_apply=False, p=0.1),   
    A.Normalize(*stats),
    ToTensor()
    ])

test_transform = A.Compose([
    A.Normalize(*stats),
    ToTensor()
   ])

# loaded only when loaddata() invoked
trainset = None
trainloader = None
testset = None
testloader = None

def __custom_getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label

class Cifar10TrainDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=train_transform)

    def __getitem__(self, index):        
        return __custom_getitem__(self, index)
  
class Cifar10TestDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data", train=False, download=True, transform=None):
        super().__init__(root=root, train=False, download=download, transform=test_transform)

    def __getitem__(self, index):
        return __custom_getitem__(self, index)
    
def loaddata():     
    global trainset, trainloader, testset, testloader, train_transform, test_transform #globals
    
    trainset = Cifar10TrainDataset()        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2) 
    testset = Cifar10TestDataset()
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2) 
    


