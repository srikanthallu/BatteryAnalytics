#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 01:12:54 2022

@author: oit
"""
import sys
import os
import argparse
import time

import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from model import IDK
from model.utils import AverageMeter
from model.utils import adjust_learning_rate, warmup_learning_rate, set_optimizer
from dataset.data import IDKDataset
from ensemble_predict import constructIDK, aggregate_idk_summary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cpu", help='cpu/gpu')
    parser.add_argument('--gpus', default=2, type=int, help='number of gpus')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    
    #model parameter
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch_size')
    
    parser.add_argument('--test_only', action="store_true", help='data is test')
    parser.add_argument('--dir_img', type=str, default='../../data/',
                        help='image directory')
    parser.add_argument('--dir_train', type=str, required=True,
                        help='train directory')
    parser.add_argument('--dir_test', required= True, 
                        help='test data directory')
    
    parser.add_argument('--dir_misclass', type=str, required=True,
                        help='pretrained model m1 location')
    parser.add_argument('--dir_m1out', required= True, 
                        help='m1 model embedding directory')
    parser.add_argument('--dir_uq', required= True, 
                        help='m1 model uncertainty value directory')
    parser.add_argument('--model_path', type=str, required=True,
                        help='idk model directory')
    parser.add_argument('--dir_out', type=str, required=True,
                        help='idk predicted result directory')
    
    
    
    args = parser.parse_args()
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

def binary_acc(y_pred, y_label):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    '''
    pos_labels= torch.nonzero(y_label.squeeze())
    correct_results_sum = (y_pred_tag[pos_labels] == y_label[pos_labels]).sum().float()
    if pos_labels.shape[0]==0:
        acc = correct_results_sum/y_label.shape[0]
    else:
        acc = correct_results_sum/pos_labels.shape[0]
    '''
    correct_results_sum = (y_pred_tag == y_label).sum().float()
    acc = correct_results_sum/y_label.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def set_model(args):
    model=IDK()
    #weights=np.array([15.75])
    #class_weights = torch.FloatTensor(weights).cuda()
    #criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    criterion = nn.BCEWithLogitsLoss()
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark=True
    #optimizer = set_optimizer(args, model)
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    
    return model, criterion, optimizer

def set_loader(args):
    train_dataset=IDKDataset(args.dir_train+args.dir_misclass, 
                             args.dir_train+args.dir_m1out, 
                             args.dir_train+args.dir_uq, is_train=True)
    train_sampler=None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle= False,
                              num_workers=args.num_workers, pin_memory=True, 
                              sampler=train_sampler)
    
    val_dataset=IDKDataset(args.dir_test+args.dir_misclass, 
                             args.dir_test+args.dir_m1out, 
                             args.dir_test+args.dir_uq, is_train=False)
    
    val_loader = DataLoader(val_dataset, batch_size=4096, 
                              shuffle= False,
                              num_workers=8, pin_memory=True)
    
    return train_loader, val_loader

def predict(args, device):
    
    test_dataset=IDKDataset(args.dir_test+args.dir_misclass, 
                             args.dir_test+args.dir_m1out, 
                             args.dir_test+args.dir_uq, is_train=False)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                              shuffle= False,
                              num_workers=8, pin_memory=True)
    
    fids= os.listdir(args.dir_img)
    map_imgid_arrid={}
    for img_id in fids:
        fname = img_id[:-4]
        map_imgid_arrid[fname] = np.zeros((224, 256))
    
    model_idk = IDK()
    state_dict = torch.load(args.model_path, map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
        state_dict = new_state_dict
       
    model_idk.to(device=device)
    model_idk.load_state_dict(state_dict)
    
    
    model_idk.eval()
    '''
    all_predictions = []
    all_labels = []
    all_pred_posx = []
    all_pred_posy = []
    all_pred_id= []
    '''
    print('data loader:',len(test_loader))
    for idx, (feat, labels, img_id, pos2d) in enumerate(test_loader):
        feat = feat.to(device=device, dtype=torch.float32)
        print('idx:',idx)    
        with torch.no_grad():
            y_pred=model_idk(feat)
            y_test_pred = torch.round(torch.sigmoid(y_pred))
            pred= y_test_pred.cpu().numpy()
        
        labels=labels.cpu().numpy()
        '''
        all_labels+=np.squeeze(labels).tolist()
        all_predictions += np.squeeze(pred).tolist()
        all_pred_id+=img_id
        all_pred_posx+=pos2d[0].tolist()
        all_pred_posy+=pos2d[1].tolist()
        '''
        labels = np.squeeze(labels).tolist()
        pos2d[0] = pos2d[0].tolist()
        pos2d[1] =pos2d[1].tolist()
        predictions = np.squeeze(pred).tolist()
        map_imgid_arrid = constructIDK(args, predictions, labels, pos2d[0], 
             pos2d[1], img_id, map_imgid_arrid)
    
    print('passing all batch:')
    #constructIDK(args, all_predictions, all_labels, all_pred_posx, 
     #        all_pred_posy, all_pred_id, map_imgid_arrid)
        
    aggregate_idk_summary(args, map_imgid_arrid)
    
    
def validate(val_loader, model, criterion, args):       
    model.eval()
    val_acc = AverageMeter()
    val_loss = AverageMeter()
    with torch.no_grad():
        for ix, (feat, labels, img_id, pos2d) in enumerate(val_loader):
            if torch.cuda.is_available():
                feat = feat.float().cuda()
                labels = labels.cuda()
            
            bsz=labels.shape[0]
            
            y_test_pred = model(feat)
            loss = criterion(y_test_pred, labels)
            acc = binary_acc(y_test_pred, labels)
            
            val_loss.update(loss.item(),bsz)
            val_acc.update(acc.item(),bsz)
            '''
            if (ix + 1) % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                  'loss {val_loss.avg:.5f} | Acc {val_acc.avg:.3f}'.format(idx+1, len(val_loader), 
                                val_loss=val_loss, val_acc=val_acc))
            sys.stdout.flush()
            '''
    return val_loss.avg, val_acc.avg

def train(args, train_loader, model, criterion, optimizer, epoch):
    model.train()
    print('training:',len(train_loader))
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()
    
    for idx, (feat, labels, img_id, pos2d) in enumerate(train_loader):
        if torch.cuda.is_available():
            feat = feat.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        bsz=labels.shape[0]
        #warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        
        y_pred = model(feat)
        loss = criterion(y_pred, labels)
        acc = binary_acc(y_pred, labels)
        
        epoch_loss.update(loss.item(), bsz)
        epoch_acc.update(acc.item(),bsz)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
              'loss {epoch_loss.avg:.5f} | Acc {epoch_acc.avg:.3f}'.format(epoch, idx + 1, 
                    len(train_loader), epoch_loss=epoch_loss, epoch_acc=epoch_acc))
            
            sys.stdout.flush()
        
    return epoch_loss.avg, epoch_acc.avg

def model_fit(args):
    train_loader, test_loader = set_loader(args)
    # build model and criterion
    model, criterion, optimizer = set_model(args)
    
    loss_file=open(args.model_path+'loss_idk.txt','w')
    loss_file.write('epoch,loss,val_loss,acc,val_acc\n')
    start=time.time()
    
    for epoch in range(args.epochs):
        #adjust_learning_rate(args, optimizer, epoch)
        loss, acc = train(args, train_loader, model, criterion, optimizer, epoch)
    
        val_loss, val_acc = validate(test_loader, model, criterion, args)
        
        print('epoch {}, val_loss {:.5f}, val_acc {:.3f}'.format(epoch, val_loss, val_acc))
        string1=str(epoch)+','+str(loss)+','+str(val_loss)+','
        string2=str(acc)+','+str(val_acc)+'\n'
        loss_file.write(string1+string2)
        
        if (epoch+1) % args.save_freq == 0:
            model_save_name=str(args.model_path+'ckpt_epoch{}.pth'.format(epoch+1))
            torch.save(model.state_dict(), model_save_name)
    
    print("Training complete in: " + str(time.time() - start))
    model_save_name=str(args.model_path+'last.pth')
    torch.save(model.state_dict(), model_save_name)
    loss_file.close()  
     
def main():
    args=parse_args()
    if args.test_only:
        if not os.path.exists(args.dir_out):
            os.makedirs(args.dir_out)
        if args.device=='cpu':
            device='cpu'
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predict(args, device)
    
    else:
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
            # build data loader
        model_fit(args)

    
if __name__ == '__main__':
    main()

