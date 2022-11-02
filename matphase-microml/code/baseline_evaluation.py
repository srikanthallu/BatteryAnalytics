#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 20:58:40 2022

@author: oit
"""

import os
import argparse

import numpy as np
import torch
from torch import optim
from PIL import Image

from metric import F1Score

import segmentation_models_pytorch as smp

import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset.data import M1Dataset


def parse_args():
    parser = argparse.ArgumentParser('baseline model evaluation')
    parser.add_argument('--topk', default=5, type=int, 
                        help='best k evaluation')
    parser.add_argument('--num_class', default=3, type=int, 
                        help='number of class')
    parser.add_argument('--in_channel', default=1, type=int, help='input channel')
    parser.add_argument('--encoder', default = 'resnet34', type= str, help='encoder name')
    parser.add_argument('--model', default = 'deeplabv3', type=str, help='model name')
    parser.add_argument('--exp', default='deeplabv3', type=str, help='experiment name')
    parser.add_argument('--dir_img', default='../../data/test_images/', type=str,  
                        help='image directory')
    parser.add_argument('--dir_label', default='../../data/test_label/', type=str,  
                        help='gt labrl directory')
    parser.add_argument('--dir_pretrain', type=str, help='model directory')
    parser.add_argument('--dir_out', type=str, required= True, 
                        help='output directory')
    
    opt = parser.parse_args()
    
    if not os.path.exists(opt.dir_out):
        os.makedirs(opt.dir_out)

    return opt


def mask_to_image(out_file,mask,num_class,save=True):
    image_arr=np.zeros((mask.shape[0],mask.shape[1]))
    cv=np.array([0,255,120])
    for c in range(num_class):
        ix_list=np.argwhere(mask==c)
        print('class ',c,': ',len(ix_list))
        
        for pos in ix_list:
            image_arr[pos[0],pos[1]]=cv[c]
    
    img=Image.fromarray(image_arr.astype(np.uint8))   
    if save:
        img.save(out_file)
    
    new_img=image_arr
    new_img[new_img==255.0]=1.0
    new_img[new_img==120.0]=2.0
    
    return img,torch.as_tensor(new_img.copy())
        
    
def predict_img(args, img_file,label_file,model,device):
    model.eval()
    img=M1Dataset.preprocess(img_file, False)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    label=torch.as_tensor(M1Dataset.preprocess(label_file, True).copy())
    label = label.to(device=device, dtype=torch.int32)
    
    with torch.no_grad():
        output = model(img)

        if args.num_class > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]
        
        prediction = probs.argmax(dim=0).cpu().numpy()
        pred_one_hot = F.one_hot(probs.argmax(dim=0),args.num_class).permute(2, 0, 1).numpy()
    
    return label, prediction, pred_one_hot

def compute_fIU(class_fIU, fid, gt, pred, label_ids):   
    misclass = np.zeros((len(label_ids),len(label_ids)))
    total =0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            l=int(gt[i][j])
            p=int(pred[i][j])
            misclass[l][p]+=1
            if l!=p:
                total+=1
    
    total_class=0
    total_pixels = gt.shape[0] *gt.shape[1]
    for l in range(len(label_ids)):
        t_miss=0
        t_ttl=0
        for m in range(len(label_ids)):
            if m!=l:
                t_miss+=misclass[m][l]
            t_ttl+=misclass[l][m]
        
        class_fIU[fid][l]=(t_ttl*misclass[l][l])/(t_ttl+t_miss)
        total_class+=class_fIU[fid][l]
        
    class_fIU[fid][len(label_ids)] = float(total_class/total_pixels)
    #print('fiu:',total_pixels,class_fIU[fid][len(label_ids)])
    return class_fIU

def compute_IU(class_IU, fid, gt, pred, label_ids):   
    misclass = np.zeros((len(label_ids),len(label_ids)))
    total =0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            l=int(gt[i][j])
            p=int(pred[i][j])
            misclass[l][p]+=1
            if l!=p:
                total+=1
    
    total_class=0
    for l in range(len(label_ids)):
        t_miss=0
        t_ttl=0
        for m in range(len(label_ids)):
            if m!=l:
                t_miss+=misclass[m][l]
            t_ttl+=misclass[l][m]
        
        class_IU[fid][l]=float(misclass[l][l]/(t_ttl+t_miss))
        total_class+=class_IU[fid][l]
    
    class_IU[fid][len(label_ids)] =float(total_class/len(label_ids))

    return class_IU

def compute_error(class_error, fid, gt, pred, label_ids):
    
    misclass = np.zeros((len(label_ids),len(label_ids)))
    total =0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            l=int(gt[i][j])
            p=int(pred[i][j])
            misclass[l][p]+=1
            if l!=p:
                total+=1
    
    for l in range(len(label_ids)):
        t_error=0
        t_ttl=0
        for m in range(len(label_ids)):
            if m!=l:
                t_error+=misclass[l][m]
            t_ttl+=misclass[l][m]
        
        class_error[fid][l]=float(t_error/t_ttl)
    
    class_error[fid][len(label_ids)] =float(total/(gt.shape[0]*gt.shape[1]))

    return class_error

def compute_accuracy(class_acc, fid, gt, pred, label_ids):
    
    all_acc = float(np.sum(np.equal(gt, pred)))
    ttl = gt.shape[0]*gt.shape[1] 
    
    for l in label_ids:
        ttl_l = np.sum(np.equal(gt,l))
        acc_l = float(np.sum(np.logical_and(np.equal(l,gt), np.equal(gt,pred))))
        class_acc[fid][l]=acc_l/ttl_l
    
    class_acc[fid][len(label_ids)] = all_acc/ttl
    
    return class_acc

def get_top_k_score(args, arr, is_reverse_sort, k):
    tarr= arr.copy()
    for l in range(args.num_class+1):
        if is_reverse_sort:
            sort_arr = np.sort(-tarr[:,l])
        else:
            sort_arr = np.sort(tarr[:,l])
        
        sort_arr = np.abs(sort_arr)
    
    topk = sort_arr[:k]
    #print('class:',l)
    #print(sort_arr[:args.topk])
    return topk
        

def main():
    args = parse_args()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    fids = os.listdir(args.dir_img)
    
    if args.model=='deeplabv3':
        model = smp.DeepLabV3(encoder_name=args.encoder, encoder_weights=None, 
                              in_channels=args.in_channel, classes=args.num_class, 
                              activation = 'softmax')
    elif args.model=='unetplusplus':
        model = smp.UnetPlusPlus(encoder_name=args.encoder, encoder_weights=None,
                           in_channels=args.in_channel, classes=args.num_class, 
                           activation = 'softmax')
    else:
        model = smp.MAnet(encoder_name=args.encoder, encoder_weights=None,
                           in_channels=args.in_channel, classes=args.num_class, 
                           activation = 'softmax')
    
    state_dict = torch.load(args.dir_pretrain, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device=device)
    test_data=M1Dataset(args.dir_img,args.dir_label)
    f1_metric = F1Score(3,'macro')
    name_list=[]
    
    class_acc = np.zeros((len(fids),4))
    class_error = np.zeros((len(fids),4))
    class_IU = np.zeros((len(fids),4))
    class_fIU = np.zeros((len(fids),4))
    class_F1 = np.zeros((len(fids),4))
    
    label_ids=[0,1,2]
    k=5
    for i,image in enumerate(fids):
        if image[-3:]!='png':
            continue
        name_list.append(image)
        
        img_file=args.dir_img+image
        label_file=args.dir_label+image
        
        true,pred,pred_one_hot = predict_img(args,img_file,label_file,model,device=device)
        np_arr_true = np.array(true)
        print('file:',image, true.shape, pred.shape)
        
        pred_image,pred_image_ts=mask_to_image(args.dir_out+image,pred,
                                               args.num_class,save=True)
        
        
        f1,prec,rec,ttl_f1 = f1_metric(pred_image_ts , true)
        class_F1[i,:3] = f1
        class_F1[i,3] = ttl_f1
        compute_error(class_error, i, np_arr_true, pred, label_ids)
        compute_accuracy(class_acc, i, np_arr_true, pred, label_ids)
        compute_IU(class_IU, i, np_arr_true, pred, label_ids)
        
        compute_fIU(class_fIU, i, np_arr_true, pred, label_ids)
        
        
        #print('f1:',ttl_f1)
    
    top_acc = get_top_k_score(args, class_acc, True, k) 
    low_acc = get_top_k_score(args, class_acc, False, k) 
    
    top_err = get_top_k_score(args, class_error, True, k)   
    low_err = get_top_k_score(args, class_error, False, k)   
    
    print('class fiu:', class_fIU[:,3])
    
    print('F1 for K', np.mean(class_F1[:,0]), np.var(class_F1[:,0]))
    print('F1 for C', np.mean(class_F1[:,1]), np.var(class_F1[:,1]))
    print('F1 for Ni', np.mean(class_F1[:,2]), np.var(class_F1[:,2]))
    print('ttl F1', np.mean(class_F1[:,3]), np.var(class_F1[:,3]))
    print('mIU', np.mean(class_IU[:,3]), np.var(class_IU[:,3]))
    
    print('fIU', np.mean(class_fIU[:,3]), np.var(class_fIU[:,3]))
    
    print('accuracy', np.mean(class_acc[:,3]), np.var(class_acc[:,3]))
    print('error', np.mean(class_error[:,3]), np.var(class_error[:,3]))
    print('top k acc', np.mean(top_acc), np.var(top_acc))
    print('lowest k acc', np.mean(low_acc), np.var(low_acc))
    print('top k error', np.mean(top_err), np.var(top_err))
    print('worst k error', np.mean(low_err), np.var(low_err))
    
    
if __name__ == '__main__':
    main()
   
    
        
        
        