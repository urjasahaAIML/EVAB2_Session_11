"""
This module defines the hyperparametres for the network
"""
dropout_value = 0.20        # not used
LR = 0.01                   # learning rate      
MOMENTUM = 0.9              
MILESTONES = [50]   # schedule for adjusting LR
GAMMA = 0.1                 # LR adjust rate
EPOCHS=24
BATCH_SIZE = 512