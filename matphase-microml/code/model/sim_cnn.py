#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:15:17 2022

@author: oit
"""

import torch
import torch.nn as nn

import os
import numpy as np
from PIL import Image
#from torch.utils.data import Dataset

class SimCNN(nn.Module):
    def __init__(self, n_channels, n_classes, loc):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.conv1 = nn.Conv2d(n_channels, 32, 3, padding =1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1) #kernel=3
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 8, 1)
        self.conv6 = nn.Conv2d(8, 8, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        fc_ch = int(loc/2)
        #self.fc1 = nn.Linear((16 * 3 * 3)+3, 128) #if use upto conv 3, conv 4
        self.fc1 = nn.Linear((8 * fc_ch * fc_ch)+3, 128) #if use conv5, conv 6
        self.batchnorm1 = nn.BatchNorm1d(128) 
        self.fc2 = nn.Linear(128, n_classes) 
        
    def forward(self, x, embed):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        #print('after pooling 1st layer:', x.shape)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x) 
        #print('after pooling 2nd layer:', x.shape)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        #print('after 3rd layer:', x.shape)
        #'''
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x1=torch.cat((x,embed),dim=1)
       # print('after flatten+concat:', x1.shape)
        x = self.dropout(self.fc1(x1))
        x = self.relu(self.batchnorm1(x))
        #print('after droput+FC1+batchnorm:', x.shape)
        x= self.fc2(x)
           
        return x

'''
def main():
    
    from torch.utils.data import DataLoader
    
    dir_img='../../../../data/validation_images/'
    dir_label='../../../../data/validation_label/'
    dir_idk='../../output/output-idk(ffn)/validation/'
    dir_embed='../../output/output-m1-idk(threshold)/validation2/prediction/'
    
    train_dataset=M2Dataset(dir_img,dir_label,dir_idk,dir_embed, 5)
    
    train_loader = DataLoader(train_dataset, batch_size=256, 
                              shuffle= False,
                              num_workers=8, pin_memory=True, 
                              sampler=None)
    
    model = SimCNN(n_channels=1, n_classes=3)
    print('train loader:', len(train_loader))
    
    for idx, (images, labels, embedding, img_id, pos2d) in enumerate(train_loader):
        pred = model(images, embedding)
        print('idx:',idx, pred.shape)
        print('\n\n')
        
        if idx>5:
            break
    
if __name__ == '__main__':
    main()
'''
