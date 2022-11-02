#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 04:38:51 2021

@author: oit
"""
import os
import glob
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

CLASSES = [0, 1, 2] #1:c, 2:Ni, 0:pore
    
class M1Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, idk_dir=None, imHW=572, maskHW=256):
        fids=os.listdir(images_dir)
        ids=[]
        for image_id in fids:
            if image_id[-3:]!='png':
                continue
            ids.append(image_id)
        self.ids=ids
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        self.idk_dir=idk_dir
        self.weight=None
        self.im_sz=imHW
        self.mask_sz=maskHW
        
    @classmethod
    def preprocess(self,imgf, is_mask):
        image = Image.open(imgf).convert('L')
        img_ndarray = np.asarray(image)
        img_ndarray=img_ndarray/255.0
        if not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]   
        else:
            nmasks=np.where((img_ndarray>0) & (img_ndarray<1.0), 2.0, img_ndarray)
            img_ndarray=nmasks
            
        image=img_ndarray
        
        return image
        
    def __len__(self): 
        return len(self.ids)
            
    def __getitem__(self, i):
        # read data
        image = Image.open(self.images_fps[i]).convert('L')
        im_dim=(self.im_sz,self.im_sz)
        
        img_ndarray = np.asarray(image)
        img_ndarray = img_ndarray[np.newaxis, ...]
        image=img_ndarray/255.0
        
        masks = Image.open(self.masks_fps[i]).convert('L')
        
        masks = np.array(masks, dtype=np.float32)
        masks=masks/255.0
        nmasks=np.where((masks>0) & (masks<1.0), 2.0, masks)
        masks=nmasks
        
        mask_shape = (len(CLASSES),masks.shape[0],masks.shape[1])
        new_masks = np.zeros(mask_shape)
        
        for c in range(len(CLASSES)):
            ix_list=np.argwhere(masks==CLASSES[c])
            for pos in ix_list:
                new_masks[c,pos[0],pos[1]]=1
        
        if self.idk_dir is None:
            idk_mask=np.ones((masks.shape[0],masks.shape[1]))
        else:
            fname=self.ids[i]
            idk_mask=np.load(self.idk_dir+fname[:-3]+'_idk.npy')
        
        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'one_hot_mask': torch.as_tensor(new_masks.copy()),
            'true_mask': torch.as_tensor(masks.copy()),
            'idk_mask': torch.as_tensor(idk_mask.copy()),
            'id' : self.ids[i]
        }

 
class IDKDataset(Dataset):
    def __init__(self, misclass_dir, out_dir, uq_dir, is_train=False):
        fids = os.listdir(misclass_dir)
        self.misclass_dir = misclass_dir
        self.out_dir = out_dir
        self.uq_dir = uq_dir
        self.pos2d=[]
        self.ids=[]
        #weights=np.zeros(2)
        for image_id in fids:
            if image_id[-3:]!='npy':
                continue
            fname=image_id[:-8]
            mis_mask = np.load(self.misclass_dir+fname+'._m1.npy')
            if is_train: #for handling large training data undersampling not misclasses
                num_pos_samples = np.count_nonzero(mis_mask)
                zero_arr = np.argwhere(mis_mask==0)
            
                neg_samples = np.random.choice(zero_arr.shape[0], 
                                           size=num_pos_samples, replace=False)
                pos_samples = np.argwhere(mis_mask==1)
                for p in neg_samples:
                    self.pos2d.append([zero_arr[p,0],zero_arr[p, 1]])
                for p in pos_samples:
                    
                    self.pos2d.append([p[0],p[1]])
                ttl_instances= num_pos_samples*2
            else:
                for i in range(mis_mask.shape[0]):
                    for j in range(mis_mask.shape[1]):
                        self.pos2d.append((i,j))
                        
                ttl_instances= mis_mask.shape[0]*mis_mask.shape[1]
            
            self.ids+=[fname]*ttl_instances
        ''' 
            w=np.count_nonzero(mis_mask)
            weights[1]+=w
            weights[0]+=(ttl_pixels-w)
        print('weights:',weights)
        '''
            
    def __len__(self): 
        return len(self.ids)       
    
    def __getitem__(self,i):
        
        fname = self.ids[i]
        mis_mask = np.load(self.misclass_dir+fname+'._m1.npy')
        out0 = np.load(self.out_dir+fname+'._out_0.npy')
        out1 = np.load(self.out_dir+fname+'._out_1.npy')
        out2 = np.load(self.out_dir+fname+'._out_1.npy')
        uq = np.load(self.uq_dir+fname+'.__m1.npy') 
        
        px = self.pos2d[i][0]
        py = self.pos2d[i][1]
        
        in_feat= torch.FloatTensor([out0[px,py],out1[px,py],out2[px,py],uq[px,py]])
        
        label = torch.tensor([mis_mask[px,py]])
        
        return in_feat, label, fname, self.pos2d[i]
    
'''          
def main():
    
    from torch.utils.data import DataLoader
    
    #dir_img='../../../data/validation_images/'
    dir_uq='../checkpoint/output-m1-idk/validation2/uq/'
    dir_mis='../checkpoint/output-m1-idk/validation2/misclass/'
    dir_out='../checkpoint/output-m1-idk/validation2/prediction/'
    
    train_dataset=IDKDataset(dir_mis, dir_out, dir_uq)
    
    train_loader = DataLoader(train_dataset, batch_size=256, 
                              shuffle= False,
                              num_workers=8, pin_memory=True, 
                              sampler=None)
    
    print('train loader:', len(train_loader))
    
    for idx, (feat, labels, img_id, pos2d) in enumerate(train_loader):
        print('idx:',idx, len(img_id), len(pos2d), pos2d[0][0].item())
       
    
        
    
if __name__ == '__main__':
    main()
'''                  
                    
                
        
        
    
    