#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:14:22 2022

Adapted from: https://github.com/HobbitLong/SupContrast
"""
from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import torch
import time
import math
from torch import optim
import torch.nn as nn

from torch.utils.data import DataLoader
#from dataset.local_data import M2Dataset
#from dataset.local_global_data import M2Dataset
from dataset.local_cl_data import M2Dataset
from torch.nn.parallel import DataParallel
import torch.backends.cudnn as cudnn

from time import sleep

from model import SimCNN
from model.utils import AverageMeter
#from model.utils import adjust_learning_rate, warmup_learning_rate
#from model.utils import set_optimizer, save_model
from model.utils import save_model

def parse_args():
    parser = argparse.ArgumentParser('argument for training M2')
    parser.add_argument('-n','--nodes', default=1,type=int, metavar='N')
    parser.add_argument('--num_class', default=3, type=int, help='number of classes')
    parser.add_argument('--gpus', default=2, type=int, help='number of gpus')
    parser.add_argument('--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--in_channel', default=1, type=int, help='input channel')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--sort', type=bool, default=True,
                        help='will data be sorted for curriculum?')
    # dataset+path
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--loc_size', type=int, default=5, 
                        help='local region hop around pixel of interest')
    parser.add_argument('--dir_img', type=str, default='../../data/', help='image directory')
    parser.add_argument('--dir_mask', type=str, default='../../data/', help='label directory')
    parser.add_argument('--dir_idk_train', type=str, required=True,
                        help='idk train directory')
    parser.add_argument('--dir_idk_test', type=str, required= True, 
                        help='idk test data directory')
    
    parser.add_argument('--dir_uq_train', type=str, 
                        default= '../output/m1-idk/train2/uq/', 
                        help='uq train data directory')
    
    parser.add_argument('--dir_uq_test', type=str, 
                        default= '../output/m1-idk/validation2/uq/', 
                        help='uq test data directory')
    
    parser.add_argument('--dir_m1out_train', type=str, required=True,
                        help='m1 output embedding directory for train')
    parser.add_argument('--dir_m1out_test', type=str, required=True,
                        help='m1 output embedding directory for test')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='checkpoint directory')
    parser.add_argument('--dir_out', type=str, required=True,
                        help='results directory')
    
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    '''
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    '''
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)
    '''
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    
    # warm-up for large-batch training,
    if args.batch_size > 256:
        args.warm = True
    if args.warm:
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    '''
    return args


def set_loader(args):
    print('Loading Train...')
    
    train_dataset=M2Dataset(args.dir_img+'train_images/',
                            args.dir_mask+'train_label/',
                            args.dir_idk_train, 
                            args.dir_m1out_train, args.dir_m1out_train, 
                            args.dir_uq_train,
                            args.loc_size, False, is_sort=args.sort)
    
    train_sampler=None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle= False,
                              num_workers=args.num_workers, pin_memory=True, 
                              sampler=train_sampler)
    
    print('Loading Validation...')
    val_dataset=M2Dataset(args.dir_img+'validation_images/',
                          args.dir_mask+'validation_label/',
                            args.dir_idk_test,
                            args.dir_m1out_test, args.dir_m1out_test, 
                            args.dir_uq_test,
                            args.loc_size, False, is_sort=False)
    
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle= False,
                              num_workers=8, pin_memory=True)
    
    #print('train loader:',len(train_loader))
    return train_loader, val_loader

def set_model(args):
    model=SimCNN(n_channels=args.in_channel, n_classes=args.num_class,
                 loc = args.loc_size)
    weights=np.array([0.86,0.63,0.52])
    class_weights = torch.FloatTensor(weights)    
    #criterion=nn.CrossEntropyLoss()
    criterion=nn.CrossEntropyLoss(weight=class_weights)
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        class_weights = class_weights.cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
        cudnn.benchmark = True

    return model, criterion

def validate(val_loader, model, criterion, args):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    val_losses = AverageMeter()
    print('validation:', len(val_loader))
    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, embed, img_id, pos2d) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                embed = embed.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # forward
            output = model(images, embed)
            loss = criterion(output, labels)

            # update metric
            val_losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            '''
            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time,loss=val_losses))
            '''
    return val_losses.avg

def train(train_loader, model, criterion, optimizer, epoch, args):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    
    for idx, (images, labels, embed, img_id, pos2d) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            embed = embed.cuda(non_blocking=True)
        bsz = labels.shape[0]
        # warm-up learning rate
        #warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images, embed)
        loss = criterion(output, labels)
        
        # update metric
        losses.update(loss.item(), bsz)

        # Adam optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
       
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
        
    return losses.avg

def main():
    args=parse_args()
    # build data loader
    train_loader, val_loader = set_loader(args)
    # build model and criterion
    model, criterion = set_model(args)
    # build optimizer
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    
    loss_file=open(args.model_path+'loss.txt','w')
    loss_file.write('epoch,loss,val_loss\n')
    start=time.time()
    for epoch in range(args.epochs):
        #adjust_learning_rate(args, optimizer, epoch)
        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, args)
        time2 = time.time()
        
        #evaluation
        val_loss = validate(val_loader, model, criterion, args)
        print('epoch {}, time {:.2f}, loss {:.2f}, val_loss {:.2f}'.format(epoch, time2 - time1, loss, val_loss))
        
        string1=str(epoch)+','+str(loss)+','+str(val_loss)+'\n'
        loss_file.write(string1)
        
        if (epoch+1) % args.save_freq == 0:
            model_save_name=str(args.model_path+'ckpt_epoch{}.pth'.format(epoch+1))
            save_model(model, optimizer, args, epoch, model_save_name)
            #torch.save(model.module.state_dict(), model_save_name)
    #save last model
    
    print("Training complete in: " + str(time.time() - start))
    model_save_name=str(args.model_path+'last.pth')
    save_model(model, optimizer, args, epoch, model_save_name)
    
    loss_file.close()
    

if __name__ == '__main__':
    main()


