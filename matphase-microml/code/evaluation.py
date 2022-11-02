#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 03:28:50 2022

@author: oit
"""
import os
import numpy as np
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser('model evaluation')
    parser.add_argument('--topk', default=5, type=int, 
                        help='best k evaluation')
    parser.add_argument('--num_class', default=3, type=int, 
                        help='number of class')
    parser.add_argument('--exp', default='chemphase', type=str, 
                        help='experiment name')
    parser.add_argument('--dir_gt', type=str, default='../../data/test_label/', 
                        help='gt directory')
    parser.add_argument('--dir_pred', type=str, required= True, 
                        help='label directory')
    parser.add_argument('--dir_out', type=str, required= True, 
                        help='output directory')
    
    opt = parser.parse_args()
    
    if not os.path.exists(opt.dir_out):
        os.makedirs(opt.dir_out)

    return opt

def preprocess_gt(args,file):
    gt_label = Image.open(args.dir_gt+file).convert('L')
    gt_label = np.asarray(gt_label)
    gt_label = gt_label/255.0
    label_ndarray=np.where((gt_label>0) & (gt_label<1.0), 2.0, gt_label)
    gt_label = label_ndarray
    
    return gt_label

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

def write_eval(args,ids,arr,eval_name):
    file = open(args.dir_out+args.exp+'_'+eval_name+'.csv','w')
    file.write('file,class,score\n')
    average =0
    for f in range(len(ids)):
        name = ids[f]
        print(name, arr[f])
        for l in range(args.num_class+1):
            file.write(name+','+str(l)+','+str(arr[f][l])+'\n')
            if l==args.num_class:
                average+=arr[f][l]
    
    average/=(len(ids)-1)
    std = np.std(arr[:,args.num_class])
    file.write('Mean,4,'+str(average)+','+str(std)+'\n')
    file.close()
    
def get_top_k_score(args, arr, fname, is_reverse_sort):
    file = open(args.dir_out+fname+'.csv','a')
    tarr= arr.copy()
    file.write('\n')
    for l in range(args.num_class+1):
        if is_reverse_sort:
            sort_arr = np.sort(-tarr[:,l])
        else:
            sort_arr = np.sort(tarr[:,l])
        
        sort_arr = np.abs(sort_arr)
        #print('class:',l)
        #print(sort_arr[:args.topk])
        file.write(args.exp+','+str(l))
        for p in range(args.topk):
            file.write(','+str(sort_arr[p]))
        
        file.write('\n')
    
    file.write('\n')
    file.close()
    
def main():
    args = parse_args()
    
    fids=os.listdir(args.dir_gt)
    class_acc = np.zeros((len(fids),4))
    class_error = np.zeros((len(fids),4))
    class_IU = np.zeros((len(fids),4))
    class_fIU = np.zeros((len(fids),4))
    label_ids =[0,1,2]
    ids=[]
    for idx in range(len(fids)):
        file = fids[idx]
        if file[-3:]!='png':
            continue
        name = file[:-4]
        ids.append(name)
        fname=args.dir_pred+name+'_ensemble.npy'
        #fname=args.dir_pred+name+'._m1.npy' #for unet, mcd-unet
        gt_label = preprocess_gt(args,file)
        pred_label = np.load(fname)
        #print('unique labels:',np.unique(pred_label), np.unique(gt_label))
        
        class_error= compute_error(class_error, idx, gt_label, pred_label, label_ids)
        class_acc= compute_accuracy(class_acc, idx, gt_label, pred_label, label_ids)
        class_IU = compute_IU(class_IU, idx, gt_label, pred_label, label_ids)
        
        class_fIU = compute_fIU(class_fIU, idx, gt_label, pred_label, label_ids)
        
        print(name,idx,class_fIU[idx,3])
        
    
    print('top k results:')
    get_top_k_score(args, class_acc, 'topk_accuracy', True)   
    get_top_k_score(args, class_error, 'worstk_error', True)   
    
    print('write results:')
    write_eval(args,ids,class_acc,'accuracy')
    write_eval(args,ids,class_error,'error')
    write_eval(args,ids,class_IU,'meanIU')
    write_eval(args,ids,class_fIU,'meanfIU')
    
    
if __name__ == '__main__':
    main()


