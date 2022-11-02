#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:06:10 2021

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
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
#from pytorch_lightning import Trainer
#from pytorch_lightning.callbacks import EarlyStopping
from tqdm import tqdm
from time import sleep

from model import UNet
#from model.utils import calculate_loss_weight

from dataset.data import M1Dataset
from losses import IDKLoss



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
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--p', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--idk_alpha', default=0.6, type=float, 
                        help='idk penalty threshold')
    
    #dataset and results
    parser.add_argument('--dir_img', type=str, default='../../data/', help='image directory')
    parser.add_argument('--dir_mask', type=str, default='../../data/', help='label directory')
    parser.add_argument('--dir_pretrain', default=None, help='pretrain directory')
    parser.add_argument('--dir_checkpoint', type=str, required=True,
                        help='checkpoint directory')
    #parser.add_argument('--dir_out', required= True, help='result directory')
    
    
    args = parser.parse_args()
    
    return args

def validate(model, dataloader, loss_fn1, loss_fn2):
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
            val_loss=loss_fn1(mask_pred, mask_true)
            val_loss+=loss_fn2(mask_pred,labels)
        
        val_losses.append(val_loss.item())
    
    return np.average(val_losses)

def set_model(args,gpu):
    model = UNet(n_channels=args.in_channel, n_classes=args.num_class, bilinear=True)
    
    model.cuda(gpu) 
    #weights=calculate_loss_weight(args.dir_mask+'train_label/')
    weights=np.array([0.96275112,0.8969029,0.14034598])
    class_weights = torch.FloatTensor(weights).cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    model = DDP(model, device_ids=[gpu])
    
    return model, optimizer, grad_scaler, class_weights

def set_data(args,rank):
    train_data=M1Dataset(args.dir_img+'train_images/',args.dir_mask+'train_label/')
    train_sampler = DistributedSampler(train_data,num_replicas=args.world_size, 
                                       rank=rank, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0, 
                              pin_memory=True, sampler=train_sampler)
    
    val_data=M1Dataset(args.dir_img+'validation_images/',args.dir_mask+'validation_label/')
    val_sampler = DistributedSampler(val_data,num_replicas=args.world_size, 
                                       rank=rank, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=0, 
                            pin_memory=True, sampler=val_sampler)
    
    return train_loader, val_loader
        
def train(gpu,args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    
    model, optimizer, grad_scaler, class_weights = set_model(args,gpu)
    
    criterion1=nn.CrossEntropyLoss(weight=class_weights).cuda(gpu)
    #criterion2=IDKLoss(alpha=args.alpha)
    #criterion2=IDKLoss(args.num_class,args.idk_alpha)
    
    train_loader, val_loader= set_data(args,rank) 
    
    num_batches=len(train_loader)*args.batch_size
    
     
    global_step=0 
    epochs=args.epochs
    
    loss_file=open(args.dir_checkpoint+'loss.txt','w')
    loss_file.write('epoch,loss,l1_loss,l2_loss,val_loss\n')
    start = datetime.now()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_loss1 = 0
        epoch_loss2 = 0
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
                    
                    loss = criterion1(pred_masks, true_masks)
                    
                    #loss2= criterion2(pred_masks, labels) 
                    #loss=loss1+loss2
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

            
                global_step += 1
                epoch_loss += loss.item()
                epoch_loss1 += loss1.item()
                epoch_loss2 += loss2.item()
                
                
            val_loss= validate(model, val_loader, criterion1, criterion2)
            tepoch.set_postfix(loss=epoch_loss/num_batches,validation=val_loss)
            sleep(0.1)
            
        if torch.distributed.get_rank() == 0:
                epoch_loss/=num_batches
                epoch_loss1/=num_batches
                epoch_loss2/=num_batches
                
                string1=str(epoch)+','+str(epoch_loss)+','+str(epoch_loss1)+','
                string2=str(epoch_loss2)+','+str(val_loss.item())+'\n'
                loss_file.write(string1+string2)
                if (epoch+1)%args.save_freq == 0:
                    model_save_name=str(args.dir_checkpoint+'ckpt_epoch{}.pth'.format(epoch + 1))
                    torch.save(model.module.state_dict(), model_save_name)
                    #torch.save(model.state_dict(), model_save_name)       
            

    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        loss_file.close()

def main():
    args=parse_args()
    if not os.path.exists(args.dir_checkpoint):
        os.makedirs(args.dir_checkpoint)
    
    args.world_size = args.gpus * args.nodes             
    os.environ['MASTER_ADDR'] = 'localhost'              
    os.environ['MASTER_PORT'] = '12355'                      
    mp.spawn(train, nprocs=args.gpus,args=(args,)) 

if __name__ == '__main__':
    main()
    
    
