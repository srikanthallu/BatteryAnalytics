#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 04:25:13 2022

@author: oit
"""
import torch.nn as nn
#import numpy as np

class IDK(nn.Module):
    def __init__(self):
        super(IDK, self).__init__()        # Number of input features is 12.
        self.layer_1 = nn.Linear(4, 16) 
        self.layer_2 = nn.Linear(16, 32)
        self.layer_out = nn.Linear(32, 1) 
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.batchnorm2 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.activation(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.activation(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
'''    
class IDK(nn.Module):
    def __init__(self, alpha, beta):
        super(IDK, self).__init__()
        
        self.alpha=alpha
        self.beta=beta
    
    def forward(self,x,uq): 
        
        #x: misclass mask during training, all ones at inference 
        #for unseen samples
        
        x1=x*uq
        out = torch.where((x1>self.alpha) & (x1<self.beta), 1, 0) #indicator function
        
        return out
'''       