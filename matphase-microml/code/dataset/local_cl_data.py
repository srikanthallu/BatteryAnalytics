#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 22:57:49 2022

@author: oit
"""


import os
import glob
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

CLASSES = [0, 1, 2] 

class M2Dataset(Dataset):
    def __init__(self, images_dir, label_dir, idk_dir, embed_dir, m1_dir, uq_dir,
                 local_HW, is_train, num_class=3, imhw=(224,256), is_sort=True):
        fids=os.listdir(images_dir)
        self.image_dir = images_dir
        self.label_dir = label_dir
        self.embed_dir = embed_dir
        self.m1_dir = m1_dir
        self.idk_dir = idk_dir
        self.uq_dir = uq_dir
        self.num_class = num_class
        
        self.hop = local_HW
        self.im_sz= 2*self.hop+1
        self.orig_im_sz=imhw
        self.poi=[]
        self.orig_image_id=[]
        self.orig_label=[]
        
        tmp_poi=[]
        tmp_label=[]
        tmp_img_id=[]
        uq=[]
        weight=np.zeros(len(CLASSES))
        num_instances=0
        for image_id in fids:
            if image_id[-3:]!='png':
                continue
            if is_train and image_id[-6:]!='_0.png':
                continue
            fname = image_id[:-4]
            mask=Image.open(self.label_dir+image_id).convert('L')
            mask = np.array(mask, dtype=np.float32)
            mask=mask/255.0
            nmasks=np.where((mask>0) & (mask<1.0), 2.0, mask)
            idk_arr=np.load(idk_dir+fname+'._idk.npy')
            uq_arr=np.load(uq_dir+fname+'.__idk.npy')
            pos=np.argwhere(idk_arr==1)
            for i in range(pos.shape[0]):
                tmp_poi.append((pos[i,0],pos[i,1]))
                tmp_img_id.append(fname)
                uq.append(uq_arr[pos[i,0],pos[i,1]])
                l=int(nmasks[pos[i,0],pos[i,1]])
                tmp_label.append(l)
                
                weight[l]+=1
                num_instances+=1
        if is_sort:
            print('before sorting len:',len(uq), len(tmp_poi),len(tmp_label))
            uq=np.array(uq)
            cl_idx= np.argsort(uq)
            for i in range(len(cl_idx)):
                self.poi.append(tmp_poi[i])
                self.orig_image_id.append(tmp_img_id[i])
                self.orig_label.append(tmp_label[i])
        
            print('after sorting len:',len(cl_idx), len(self.poi),len(self.orig_label))
        else:
            self.poi = tmp_poi
            self.orig_image_id = tmp_img_id
            self.orig_label = tmp_label
        
        '''
        weight/=num_instances
        self.weight=1-weight
        print(self.weight,num_instances)
        '''
           
    def __len__(self):
        return len(self.poi)
    
    def __getitem__(self, index):
        fname = self.orig_image_id[index] 
        image = Image.open(self.image_dir+fname+'.png').convert('L')
        img_ndarray = np.asarray(image)
        image=img_ndarray/255.0
        image=np.asarray(image)
        labels=torch.as_tensor(self.orig_label[index])
        
        idk_arr = np.load(self.idk_dir+fname+'._idk.npy')
        pred_m1 = np.load(self.m1_dir+fname+'._m1.npy')
        
        #im_dim=(self.im_sz,self.im_sz)
        #local_image= np.zeros(im_dim)
        input_channel = np.zeros((self.num_class+1, self.im_sz,self.im_sz))
        poi_pos=self.poi[index]
        
        pixel_center=(self.hop,self.hop)#(2,2)
        for i in range(self.im_sz):
            hopx=i-pixel_center[0]
            x=poi_pos[0]+hopx
            if x<0 or x>=self.orig_im_sz[0]:
                continue
            for j in range(self.im_sz):
                hopy=j-pixel_center[1]
                y=poi_pos[1]+hopy
                if y<0 or y>=self.orig_im_sz[1]:
                    continue
                input_channel[0,i,j]=image[x,y]
                if idk_arr[x,y]==0:
                    m1_label = pred_m1[x,y]
                    input_channel[m1_label+1,i,j]=1
        
        input_channel[0] = (input_channel[0]-0.5)/0.5
        '''
        local_image=Image.fromarray(input_channel[0], 'L')
        
        '''
        images=torch.as_tensor(input_channel).float()#.contiguous()
        
        e1 = np.load(self.embed_dir+fname+'._out_0.npy')
        e2 = np.load(self.embed_dir+fname+'._out_1.npy')
        e3 = np.load(self.embed_dir+fname+'._out_2.npy')
        px,py=self.poi[index][0],self.poi[index][1]
            
        embedding = torch.FloatTensor([e1[px,py], e2[px,py], e3[px,py]])
        #print('p,c,ni:',np.count_nonzero(input_channel[1]),
         #     np.count_nonzero(input_channel[2]),
          #    np.count_nonzero(input_channel[3]),labels.item())
        
        return images,labels, embedding, self.orig_image_id[index],self.poi[index]

'''
def main():
    from torch.utils.data import DataLoader
    dir_img='../../../../data/test_images/'
    dir_label='../../../../data/test_label/'
    dir_idk='../../output/output-idk(ffn)/test/'
    dir_m1='../../output/output-m1-idk(threshold)/test2/prediction/'
    dir_uq='../../output/output-m1-idk(threshold)/test2/uq/'
    dir_embed = '../../output/output-m1-idk(threshold)/test2/prediction/'
    train_dataset=M2Dataset(dir_img,dir_label,dir_idk,dir_embed, dir_m1, dir_uq, 5, True)
    
    train_loader = DataLoader(train_dataset, batch_size=256, 
                              shuffle= False,
                              num_workers=0, pin_memory=True, 
                              sampler=None)
    
    print('loader:',len(train_loader))
    for idx, (images, labels, embedding, img_id, pos2d) in enumerate(train_loader):
        #print('idx:',idx,torch.unique(labels),torch.bincount(labels))
        print('idx:',idx, images.shape)
        if idx>5:
            break
    
if __name__ == '__main__':
    main()
'''
