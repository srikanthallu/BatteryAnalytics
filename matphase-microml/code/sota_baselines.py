#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:42:06 2022

@author: oit
"""


import os
import argparse
from datetime import datetime
#import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

import segmentation_models_pytorch as smp

from tqdm import tqdm
from time import sleep


from dataset.data import M1Dataset



classes=[0,1,2]

def parse_args():
    parser = argparse.ArgumentParser('argument for training M1')
    parser.add_argument('-n','--nodes', default=1,type=int, metavar='N')
    parser.add_argument('--num_class', default=3, type=int, help='number of classes')
    parser.add_argument('--gpus', default=2, type=int, help='number of gpus')
    parser.add_argument('--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--save_freq', default=20, type=int, 
                        help='freq to save model')
    parser.add_argument('--in_channel', default=1, type=int, help='input channel')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=30, type=int, help='batch size')
    parser.add_argument('--encoder', default = 'resnet34', type= str, help='encoder name')
    parser.add_argument('--model', default = 'deeplabv3', type= str, 
                        help='choose model name deeplab/unet+/manet')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    
    #dataset and results
    parser.add_argument('--dir_img', type=str, default='../../data/', help='image directory')
    parser.add_argument('--dir_mask', type=str, default='../../data/', help='label directory')
    parser.add_argument('--dir_checkpoint', type=str, required=True,
                        help='checkpoint directory')
    #parser.add_argument('--dir_out', required= True, help='result directory')
    
    
    args = parser.parse_args()
    
    return args

def validate(model, dataloader, loss_fn):
    model.eval()
    #num_batches = len(dataloader)
    val_losses = []

    for batch in dataloader:
        image, mask_true, labels = batch['image'], batch['one_hot_mask'], batch['true_mask']
        
        image = image.cuda(non_blocking=True)
        mask_true = mask_true.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with torch.no_grad():
            mask_pred = model(image)
            val_loss=loss_fn(mask_pred, mask_true)
            
        val_losses.append(val_loss.item())
    
    return np.average(val_losses)

def set_model(args,device):
    enc_weights = None
    if args.model=='deeplabv3':
        model = smp.DeepLabV3(encoder_name=args.encoder, encoder_weights=enc_weights, 
                              in_channels=args.in_channel, classes=args.num_class, activation = 'softmax')
    elif args.model=='unetplusplus':
        model = smp.UnetPlusPlus(encoder_name=args.encoder, encoder_weights=enc_weights,
                           in_channels=args.in_channel, classes=args.num_class, activation = 'softmax')
    else:
        model = smp.MAnet(encoder_name=args.encoder, encoder_weights=enc_weights,
                           in_channels=args.in_channel, classes=args.num_class, activation = 'softmax')
    
    weights=np.array([0.96275112,0.8969029,0.14034598])
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        class_weights = torch.FloatTensor(weights).cuda()
        class_weights = class_weights.cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    
    
    #preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, enc_weights)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    
    return model, optimizer, grad_scaler, class_weights, criterion

def set_data(args):
    train_data=M1Dataset(args.dir_img+'train_images/',args.dir_mask+'train_label/')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0, 
                              pin_memory=True, shuffle=False)
    
    val_data=M1Dataset(args.dir_img+'validation_images/',args.dir_mask+'validation_label/')
    
    val_loader = DataLoader(val_data, batch_size=1, num_workers=0, pin_memory=True, shuffle=False)
    
    return train_loader, val_loader
        
def train(device,args):
    
    model, optimizer, grad_scaler, class_weights, criterion = set_model(args,device)
    
    train_loader, val_loader= set_data(args) 
    
    num_batches=len(train_loader)
    
     
    global_step=0 
    epochs=args.epochs
    
    loss_file=open(args.dir_checkpoint+'loss.txt','w')
    loss_file.write('epoch,loss,val_loss\n')
    start = datetime.now()
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for batch in tepoch:
                if torch.cuda.is_available():
                    images=batch['image'].cuda(non_blocking=True)
                    true_masks=batch['one_hot_mask'].cuda(non_blocking=True)
                    labels=batch['true_mask'].cuda(non_blocking=True)
                
                tepoch.set_description(f"Epoch {epoch}")
                
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=False):
                    pred_masks = model(images)
                    
                    loss = criterion(pred_masks, labels)
                    
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

            
                global_step += 1
                epoch_loss += loss.item()
                
            val_loss= validate(model, val_loader, criterion)
            tepoch.set_postfix(loss=epoch_loss/num_batches,validation=val_loss)
            sleep(0.1)
            
            epoch_loss/=num_batches
                
            string1=str(epoch)+','+str(epoch_loss)+','+str(val_loss.item())+'\n'
            loss_file.write(string1)
                
            if (epoch+1)%args.save_freq == 0:
                model_save_name=str(args.dir_checkpoint+'ckpt_epoch{}.pth'.format(epoch + 1))
                torch.save(model.module.state_dict(), model_save_name)
                    #torch.save(model.state_dict(), model_save_name)       
            
    
    print("Training complete in: " + str(datetime.now() - start))
    torch.save(model.module.state_dict(), 'last.pth')
    loss_file.close()

def main():
    args=parse_args()
    if not os.path.exists(args.dir_checkpoint):
        os.makedirs(args.dir_checkpoint)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device,args)

if __name__ == '__main__':
    main()
    
    
