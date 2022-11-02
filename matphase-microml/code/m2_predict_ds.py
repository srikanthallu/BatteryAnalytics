#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 22:43:53 2022
Linear Evaluation Stage
Adapted from: https://github.com/HobbitLong/SupContrast
"""
from __future__ import print_function
import os
import sys
import argparse
import time
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
#from dataset.local_data import M2Dataset
from dataset.data import M1Dataset
#from dataset.local_global_data import M2Dataset
from dataset.local_cl_data import M2Dataset

import torch.nn.functional as F
from torch.nn.parallel import DataParallel

from model import SimCNN, UNet
from model.deep_sup import DeepSup

from model.utils import write_f1_lines
from metric import F1Score
from ensemble_predict import aggregate_ensemble_summary
from ensemble_predict import construct_ensemble, get_global_labels
      

def parse_args():
    parser = argparse.ArgumentParser()
    
    #hyperparameters
    parser.add_argument('--num_class', default=3, type=int, 
                        help='number of classes')
    parser.add_argument('--in_channel', default=1, type=int, 
                        help='input channel')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--test_only', action="store_true", help='data is test')
    parser.add_argument('--device', default="cpu", help='cpu/gpu')
    parser.add_argument('--num_workers', default=1, type = int,
                        help='num workers')
    
    #parameter
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--loc_size', type=int, default=5, 
                        help='local region hop around pixel of interest')
    
    #model+output directory
    parser.add_argument('--dir_img', type=str, default='../../data/test_images/',
                        help='image directory')
    parser.add_argument('--dir_mask', type=str, default='../../data/test_label/', 
                        help='label directory')
    parser.add_argument('--dir_uq', type=str, 
                        default='../output/m1/test2/uq/',
                        help='uq directory')
    parser.add_argument('--dir_pretrain_m2', type=str, required=True,
                        help='file-name pretrained model m2')
    parser.add_argument('--dir_idk', type=str, required=True,
                        help='idk directory from model IDK')
    parser.add_argument('--dir_outm1', type=str, required=True,
                        help='directory of m1 embedding')
    parser.add_argument('--dir_m1', type=str, required=True,
                        help='directory of m1 predictions')
    parser.add_argument('--dir_out', required= True, help='result directory')
    
    args = parser.parse_args()
    
    return args

def calculate_misclass(all_pred, all_labels, args):
    print('M2 misclass scores:')
    mis_file=open(args.dir_out+'misclass_m2{}.txt'.format(args.sample_type),'w')
    mis=np.zeros((3,3))
    num_ins=len(all_labels)
    
    for i in range(len(all_labels)):
        l=int(all_labels[i])
        p=int(all_pred[i])
        mis[l][p]+=1
    mis/=num_ins
    mis_file.write('true,predicted,score\n')
    for i in range(args.num_class): 
        for j in range(args.num_class):
            line=str(i)+','+str(j)+','+str(mis[i][j])+'\n'
            mis_file.write(line)
    
    mis_file.write('total instances:'+str(num_ins)+'\n')
    mis_file.close()
    
def set_model(args,device):
    if args.model == 'cnn':
        model_m2 = SimCNN(n_channels=args.in_channel, n_classes=args.num_class, 
                      loc = args.loc_size)
    
    else:
        model_m2 = DeepSup(n_channels=args.in_channel, n_classes=args.num_class, 
                      loc = args.loc_size)
    
    ckpt = torch.load(args.dir_pretrain_m2, map_location=device)
    
    state_dict = ckpt['model']
    
    new_state_dict = {}
    
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
        state_dict = new_state_dict
    
    model_m2.to(device=device)
    model_m2.load_state_dict(state_dict)
    
    return model_m2

def set_loader(args):
    test_dataset=M2Dataset(args.dir_img, args.dir_mask, 
                           args.dir_idk, args.dir_outm1, args.dir_outm1, 
                           args.dir_uq, args.loc_size, 
                           False, is_sort=False)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle= False, num_workers=args.num_workers, 
                             pin_memory=True)
    
    return test_loader

def predict_cnn(f1_file, model_m2, test_loader, args, device):
    model_m2.eval()
    
    #map_imgid_ensemble = get_global_labels(args)
    print('test loader:',len(test_loader))
    f1_metric = F1Score(3,'macro')
    for idx, (images, labels, embed, img_id, pos2d) in enumerate(test_loader):
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device)
        embed = embed.to(device=device)
        with torch.no_grad():
            pred1, pred2, pred_final=model_m2(images, embed)
            
            probs = F.softmax(pred_final, dim=1)
            pred_labels=probs.argmax(dim=1)
                    
        f1,prec,rec,ttl_f1 = f1_metric(pred_labels, labels)
        f1_file.write(write_f1_lines('batch_{}'.format(str(idx)),prec,rec,f1,ttl_f1))
        '''
        predictions=list(pred_labels.numpy())#pred_labels.cpu().numpy()
        labels=list(labels.numpy())#labels.cpu().numpy()
        map_imgid_ensemble = construct_ensemble(args, predictions, labels, 
                            pos2d[0], pos2d[1], img_id, map_imgid_ensemble)
        '''
        print('idx:',idx, f1)
    
    #print('emsemble M1+M2 predictions:')
    #aggregate_ensemble_summary(args, map_imgid_ensemble)
    
    
def m2_predict(args, device):
    f1_file=open(args.dir_out+'f1_m2_{}_{}.txt'.format(args.model, 
                                                       args.sample_type),'w')
    
    model_m2 = set_model(args,device)
    test_loader = set_loader(args)
    predict_cnn(f1_file, model_m2, test_loader, args, device)
    

    f1_file.close()
    
    
    
def main():
    
    args=parse_args()

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)
    
    if args.test_only:
       args.sample_type='test'
    else:
        args.sample_type='validation'
    
    if args.device=='cpu':
        device='cpu'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    m2_predict(args,device)


if __name__ == '__main__':
    main()





