"""
This module contains utilities to display sample images.
Can be used for additional functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import dataloader
from dataloader import *
from testing import *
from training import *

# display some training images in a grid
def displaysampleimage():    
    _dataiter = iter(dataloader.trainloader)
    _images, _labels = _dataiter.next()
    print('shape of images', _images.shape)
    _sample_images = _images[0:4,:,:,:] # first 4 images
   
    # show images
    __imshow__(torchvision.utils.make_grid(_sample_images))
    # print labels
    print(' '.join('%5s' % classes[_labels[j]] for j in range(4)))
    

# diaplay an image
def __imshow__(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    

# display 25 misclassified images in a grid
def displaymisclassified():
    fig = plt.figure(figsize=(15,15))
    plt.title("Misclassified images\n\n")
    plt.axis("off")   
    for index in range(25):
        ax = fig.add_subplot(5, 5, index + 1, xticks=[], yticks=[])
        image = misclassifiedImages[index] 
        npimg = np.transpose(image.cpu().numpy(), (1, 2, 0))
        pred = misclassifiedPredictions[index].cpu().numpy()
        target = misclassifiedTargets[index].cpu().numpy()
        ax.imshow(npimg.squeeze())
        ax.set_title(f'pred:{classes[pred]},target={classes[target]}')
        plt.subplots_adjust(wspace=1, hspace=1.5)
    plt.show()
    
# display classwise test accuracy %
def displayaccuracybyclass():
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


def plot_CLR_graph(model):
    
    ''' Plot Cycle LR graph with LR_Min=0.001 , LR_Max=1.0'''
    
    _optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    _scheduler = torch.optim.lr_scheduler.CyclicLR(_optimizer, base_lr=0.001, max_lr=1.0, step_size_up=20,mode="triangular")
    lrs = []
    for i in range(120):
        _optimizer.step()
        lrs.append(_optimizer.param_groups[0]["lr"])        
        _scheduler.step()
    plt.ylabel("LR")
    plt.xlabel("Iterations")
    plt.title("Traingular Cycle LR\n\n")
    plt.plot(lrs)
    
    
def plot_OneCycle_graph(model):
    
    ''' Plot One Cycle LR graph with my LRMax'''
    
    _optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    _scheduler = torch.optim.lr_scheduler.OneCycleLR(_optimizer, max_lr=0.8, pct_start=0.2, steps_per_epoch=int(50000/512), epochs=24, anneal_strategy='linear')
    lrs = []
    for i in range(2328):
        _optimizer.step()
        lrs.append(_optimizer.param_groups[0]["lr"])       
        _scheduler.step()
        
    plt.ylabel("LR")
    plt.xlabel("Iterations")
    plt.title("One Cycle LR\n\n")
    plt.plot(lrs)
    
def plotLRRangeTest_graph(num_epochs):
    lrs = np.arange (0.001, 1.2, 0.1)

    for lr in lrs:
        definelossfunction_with_LR(lr)
        model = trainmodel(num_epochs)  # 3 epochs only

    #print(lrs, test_accuracies_for_LRRageTest)
    fig, ax = plt.subplots()
    ax.plot(lrs[0:12], test_accuracies_for_LRRageTest[0:12])
    ax.set(xlabel='Learning Rate', ylabel='Test Accuracy', title='test')
    ax.grid()
    plt.show()
    

def plotAccuracy():
    global train_accuracies, test_accuracies
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    axs.set_title('Train/Validation Accuracy')
    axs.set_xlabel('epoch')
    axs.set_ylabel('% accuracy')
    
    p1, = plt.plot(train_accuracies,label='Training Accuracy')
    p2, = plt.plot(test_accuracies, label='Validation Accuracy')
  

    plt.legend(handles=[p1, p2], title='Train/ Validation Accuracy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()