"""
Main module a.k.a control module.
This module can orchesterate the flow to train and test the network.
"""
import hyperparameters
from hyperparameters import * 
import model_resnet
from model_resnet import * 
import dataloader
from dataloader import *
import training
from training import *
import testing
from testing import *
import utils
from utils import *

#loaddata()              # load data from CIFAR10 dataset
#displaysampleimage()    # display sample images
#createmodel()           # crease Resnet18 model
#modelSummary()          # print model summary for analysis
#definelossfunction()    # define optimizer, scheduler, loss funcrion
#trainmodel()            # train model
#testmodel()             # test model
#stats_classified()      # misclassified images