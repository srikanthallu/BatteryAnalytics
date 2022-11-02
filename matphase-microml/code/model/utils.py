#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 04:53:03 2021

@author: oit
"""
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def write_f1_lines(fimage,prec,rec,f1,ttl_f1):
    str0='file,'+fimage+'\n'
    str1='0,'+str(prec[0])+','+str(rec[0])+','+str(f1[0])+'\n'
    str2='1,'+str(prec[1])+','+str(rec[1])+','+str(f1[1])+'\n'
    str3='2,'+str(prec[2])+','+str(rec[2])+','+str(f1[2])+'\n'
    str4='ttl-f1,'+str(ttl_f1)+'\n'
    
    return str0+str1+str2+str3+str4

def check_li_percentage(indir,outdir,name_list):
    num_c=0
    num_k=0
    num_ni=0
    total=0
    for name in name_list:
        file=indir+name
        img=Image.open(file).convert('L')
        np_arr=np.array(img)
        np_arr=np_arr/255.0
        pixel=int(np_arr.shape[0]*np_arr.shape[1])
        no_c=np.count_nonzero(np_arr == 1.0)
        no_pore=np.count_nonzero(np_arr == 0)
        no_ni=pixel-(no_c+no_pore)
        
        
        num_c+=no_c
        num_ni+=no_ni
        num_k+=no_pore
        total+=pixel
        
        print('ttl,c,ni,pore',name,float(no_c/pixel),float(no_ni/pixel),float(no_pore/pixel))
        
    print('total:',float(num_c/total),float(num_ni/total),float(num_k/total))

def mask_to_image(out_file,mask,num_classes,save=False):
    #print('mask shape:',mask.shape)
    image_arr=np.zeros((mask.shape[1],mask.shape[2]))
    cv=np.array([0,255,120])
    for c in range(num_classes):
        ix_list=np.argwhere(mask[c]==1)
        #print('class ',c,': ',len(ix_list))
        for pos in ix_list:
            image_arr[pos[0],pos[1]]=cv[c]
            
    img=Image.fromarray(image_arr.astype(np.uint8))   
    if save:
        img.save(out_file)
    
    new_img=image_arr
    new_img[new_img==255.0]=1.0
    new_img[new_img==120.0]=2.0
    
    return img,torch.as_tensor(new_img.copy())
        
    
def calculate_loss_weight(mask_dir,num_class=3,idk_dir=None):
        dir_mask=mask_dir+'train_label/'
        ids = os.listdir(dir_mask)
        weight=np.zeros(num_class)
        
        for ix in range(0,len(ids)):
            name=ids[ix]
            if name[-3:]!='png':
                continue
            img=Image.open(dir_mask+name).convert('L')
            np_arr=np.asarray(img)
            np_arr=np_arr/255.0
            if idk_dir is not None:
                idk_mask=np.load(idk_dir+name[:-3]+'_idk.npy')
            else:
                idk_mask=np.ones(np_arr.shape)
            pixel=0
            tmp_weight=np.zeros(3)
            for i in range(np_arr.shape[0]):
                for j in range(np_arr.shape[1]):
                    if np_arr[i][j]>0 and np_arr[i][j]<1:
                        label=2
                    else:
                        label=int(np_arr[i][j])
                    if idk_mask[i][j]:
                        tmp_weight[label]+=1
                        pixel+=1
            tmp_weight=tmp_weight/pixel
        
            for i in range(num_class):
                weight[i]+=(1-tmp_weight[i])
        
        weight/=len(ids)
        #print('weight:',weight)
        return weight
    
def plot_mask_one_hot(img_name, orig_mask, masks_one_hot):
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(2, 2)
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[1, 0]
        ax4 = axs[1, 1]
        
        ax1.spy(orig_mask)
        ax1.set_title('original')
        ax2.spy(masks_one_hot[0,:,:],markersize=5)
        ax2.set_title('O2 one hot vector')
        ax3.spy(masks_one_hot[1,:,:],markersize=5)
        ax3.set_title('C one hot vector')
        ax4.spy(masks_one_hot[2,:,:],markersize=5)
        ax4.set_title('Ni one hot vector')
        
        plt.show()

def print_misclass(misclass,num_c):
    #print('misclass:')
    lines=[]
    for i in range(num_c):
        for j in range(num_c):
            line=','+str(i)+','+str(j)+','+str(misclass[i][j])+'\n'
            lines.append(line)
            #print(line)
    return lines
       
def plot_misclassification(label, prediction, label_ids):
    #print('label values:', np.unique(label),np.unique(prediction))
    c=1
    o=0
    ni=2
    
    color_map={}
    for k in label_ids:
        color_map[k]={}
    
    
    #color_map[c][ni]=[255,255,0] #Y
    color_map[c][o]=[255,0,0] #R
    color_map[c][c]=[255,255,255] #W
    
    #color_map[ni][c]=[0,0,255] #B
    color_map[ni][o]=[0,255,255] #sky
    color_map[ni][ni]=[128,128,128] #Gray
    
    #color_map[o][ni]=[0,255,0] #G
    color_map[o][c]=[0,128,0] #Lime
    color_map[o][o]=[0,0,0] #K/black
    
    misclass=np.zeros((3,3))
    
    new_img_array=np.zeros((label.shape[0],label.shape[1],len(label_ids)))
    misclass_mask=np.zeros((label.shape[0],label.shape[1]))
        
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            l=int(label[i][j])
            p=int(prediction[i][j])
            if l==p:
                new_img_array[i][j]=color_map[l][p]
                misclass[l][l]+=1
            else:
                misclass_mask[i][j]=1
                misclass[l][p]+=1
                if l!=0:
                    new_img_array[i][j]=color_map[l][0]
                else:
                    new_img_array[i][j]=color_map[l][1]
                
    
    img = Image.fromarray(np.uint8(new_img_array))
    lines=print_misclass(misclass,3)
    
    return lines,img,misclass_mask
    
    
def plot_img_and_mask(outfile,img_file, label_file, predict, misclass):
    label = Image.open(label_file).convert('L')
    label = np.asarray(label)
    
    in_img=Image.open(img_file)
    in_img=np.asarray(in_img)
    
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 4, 1)
    #fig, ax = plt.subplots(1, classes + 1)
    plt.title('Input')
    plt.imshow(in_img,cmap='gray')
    
    plt.subplot(1, 4, 2)
    plt.title('GT')
    plt.imshow(label,cmap='gray')
    
    plt.subplot(1, 4, 3)
    plt.title('Predicted')
    plt.imshow(predict,cmap='gray')
     
    plt.subplot(1, 4, 4)
    plt.title('misclass')
    plt.imshow(misclass)
    
    plt.xticks([])
    plt.yticks([])
    plt.savefig(outfile)
    #plt.show()
    #plt.cla()
    #plt.close()

