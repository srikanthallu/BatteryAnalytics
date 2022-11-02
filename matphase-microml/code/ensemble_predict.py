#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:37:55 2022

Aggregated test Predictions from M1-->IDK--> M2 (Local Model--> CNN)
@author: oit
"""

import os

import numpy as np
import torch
from torch import optim
from PIL import Image

from metric import F1Score
from model.utils import write_f1_lines, plot_misclassification
from model.utils import plot_img_and_mask

def print_misclass(misclass_file, fname, misclass_summ):
    str0=fname+',,,\n'
    misclass_file.write(str0)
    for l in misclass_summ:
        misclass_file.write(l)
    
def convert_pred_to_image(pred):
    image_arr=pred
    image_arr[image_arr==1]=255
    image_arr[image_arr==2]=120
    return image_arr

def aggregate_ensemble_summary(args, map_imgid_ensemble):
    result_dir=args.dir_out+'ensemble/'
    mis_dir = args.dir_out+'misclass/'
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    if not os.path.exists(mis_dir):
        os.makedirs(mis_dir)
    
    f1_file=open(args.dir_out+'f1_ensemble.txt','w')
    misclass_file=open(args.dir_out+'misclass_ensemble.txt','w')
    fids=os.listdir(args.dir_img)
    
    f1_metric = F1Score(3,'macro')
    for image_id in fids:
        if image_id[-3:]!='png':
                continue
        fname=image_id[:-4]
        mask=Image.open(args.dir_mask+image_id).convert('L')
        gt = np.array(mask, dtype=np.float32)
        gt=gt/255.0
        tmp=np.where((gt>0) & (gt<1.0), 2.0, gt)
        gt=tmp
        ensemble_pred = map_imgid_ensemble[fname]
        
        lines,misclass,_=plot_misclassification(gt, ensemble_pred, [0,1,2])
        
        f1,prec,rec,ttl_f1 = f1_metric(torch.FloatTensor(ensemble_pred), 
                                       torch.FloatTensor(gt))
        
        f1_file.write(write_f1_lines(image_id,prec,rec,f1,ttl_f1))
        
        np.save(result_dir+fname+'_ensemble',ensemble_pred)
        
        print_misclass(misclass_file, fname, lines)
        
        
        img_file=args.dir_img+image_id
        label_file=args.dir_mask+image_id
        ensemble_img=convert_pred_to_image(ensemble_pred)
        plot_img_and_mask(result_dir+image_id,img_file,label_file,
                          ensemble_img,misclass)
        
        misclass.save(mis_dir+fname+'_m1.png')
        
    f1_file.close()
        
        
def get_global_labels(args, is_m1=True):
    fids=os.listdir(args.dir_img)
    map_imgid_ensemble = {}
    for image_id in fids:
        if image_id[-3:]!='png':
                continue
        fname=image_id[:-4]
        m1_pred=np.load(args.dir_outm1+fname+'._m1.npy')
        if is_m1:
            map_imgid_ensemble[fname] = m1_pred 
        else: #this condition is for adapeted-lcgar predictions
            map_imgid_ensemble[fname] = np.zeros(m1_pred.shape)
    
    return map_imgid_ensemble

def construct_ensemble(args, predictions, labels, posx, posy, img_id, map_imgid_ensemble):
    for idx in range(len(img_id)):
        x, y= posx[idx], posy[idx]
        map_imgid_ensemble[img_id[idx]][x, y] = predictions[idx]
    
    return map_imgid_ensemble 

def get_idk_summary(idk_mask,misclass_mask):
    tp_idk,mis_idk=0,0
    tp_nidk,mis_nidk=0,0
    
    mis_idk = np.count_nonzero(idk_mask*misclass_mask)
    mis_nidk = np.count_nonzero((1-idk_mask)*misclass_mask)
    tp_idk = np.count_nonzero(idk_mask*(1-misclass_mask))
    tp_nidk = np.count_nonzero((1-idk_mask)*(1-misclass_mask))
    
    print('idk-with-miss:',mis_idk,'idk-tp:',tp_idk)
    
    
    mis_idk/=np.count_nonzero(misclass_mask)
    mis_nidk/=np.count_nonzero(misclass_mask)
    tp_idk/=np.count_nonzero((1-misclass_mask))
    tp_nidk/=np.count_nonzero((1-misclass_mask))
    
    string=str(tp_idk)+','+str(tp_nidk)+','+str(mis_idk)+','+str(mis_nidk)+'\n'
    
    
    return string

def constructIDK(args, predictions, labels, posx, posy, img_id, 
                  map_imgid_arrid):
    
    for idx in range(len(img_id)):
        x, y= posx[idx], posy[idx]
        map_imgid_arrid[img_id[idx]][x, y] = predictions[idx]
    
    return map_imgid_arrid

def aggregate_idk_summary(args, map_imgid_arrid):
    
    fids=os.listdir(args.dir_img)
    idk_file=open(args.dir_out+'idk_summary.txt','w')
    idk_file.write('file,tp-in-idk,tp-not-idk,misclass-idk,misclass-nidk\n')
    
    for image_id in fids:
        if image_id[-3:]!='png':
            continue
        fname=image_id[:-4]
        misclass = np.load(args.dir_test+args.dir_misclass+fname+'._m1.npy')
        idk_mat = map_imgid_arrid[fname]
        print('constructing IDK for ', fname, np.count_nonzero(misclass),
              np.count_nonzero(idk_mat))
       
        line = get_idk_summary(idk_mat,misclass) 
        
        idk_file.write(fname+','+line)
        np.save(args.dir_out+fname+'._idk.npy',idk_mat)

    idk_file.close()

        
        
        
        
        
       
        
        
        
    
