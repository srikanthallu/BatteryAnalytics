#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:02:34 2021

@author: oit
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from dataset.data import M1Dataset
from model import UNet
#from model import IDK
from model.utils import plot_misclassification, plot_img_and_mask
from model.utils import mask_to_image, write_f1_lines
from metric import F1Score

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--idk_th_min', default=0.45, type=float, 
     #                   help='uq threshold for idk classifier')
    #parser.add_argument('--idk_th_max', default=0.55, type=float, 
     #                   help='uq threshold for idk classifier')
    parser.add_argument('--Ntest', default=20, type=int,
                        help='number predictions for uncertainty')
    parser.add_argument('--num_class', default=3, type=int, help='number of classes')
    parser.add_argument('--p', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--test_only', action="store_true", help='data is test')
    parser.add_argument('--device', default="cpu", help='cpu/gpu')
    parser.add_argument('--dir_img', type=str, default='../../data/train_images/',
                        help='image directory')
    parser.add_argument('--dir_mask', type=str, default='../../data/train_label/', 
                        help='label directory')
    parser.add_argument('--dir_pretrain', type=str, required=True,
                        help='pretrained model m1 location')
    parser.add_argument('--dir_out', required= True, help='result directory')
    
    args = parser.parse_args()
    
    return args

def save_files(fname,misclass_file,f1_file,f1_summ, misclass_summ):
    
    f1_file.write(write_f1_lines(fname,f1_summ[0],f1_summ[1],f1_summ[2],f1_summ[3]))
    #f1nidk_file.write(write_f1_lines(fname,f1nidk_summ[0],f1nidk_summ[1],f1nidk_summ[2],
    #                                 f1nidk_summ[3]))
    str0=fname+',,,\n'
    misclass_file.write(str0)
    for l in misclass_summ:
        misclass_file.write(l)
    
    #idk_file.write(fname+','+idk_summ)
    
def save_predictions(args,sample_type,fname,pred_mask_m1,uq_m1,
                     misclass,misclass_mask, out):
    '''
    mis_uq=misclass_mask*uq_m1.numpy()
    tp_uq=(1-misclass_mask)*uq_m1.numpy()
    m1=mis_uq.ravel()[np.flatnonzero(mis_uq)]
    m2=tp_uq.ravel()[np.flatnonzero(tp_uq)]
    
    print('misclasses:',np.count_nonzero(misclass_mask),
          'idk:',torch.count_nonzero(pred_idk).item(),
          'misclass uq values:',np.min(m1),np.max(m1),np.mean(m1),
          'tp uq values:',np.min(m2),np.max(m2), np.mean(m2))
    '''
    pred_dir=args.dir_out+'prediction/'+fname+'_'
    #idk_dir=args.dir_out+'idk-'+sample_type+'/'+fname+'_'
    np.save(pred_dir+'m1',pred_mask_m1.cpu().numpy())
    #np.save(idk_dir+'idk',pred_idk.numpy())
    np.save(pred_dir+'out_0',out[0])
    np.save(pred_dir+'out_1',out[1])
    np.save(pred_dir+'out_2',out[2])
    
    mis_dir=args.dir_out+'misclass/'+fname+'_'
    np.save(mis_dir+'m1',misclass_mask)
    misclass.save(mis_dir+'m1.png')
    
    uq_dir=args.dir_out+'uq/'+fname+'_'
    np.save(uq_dir+'m1',uq_m1.numpy()) 
   #np.save(uq_dir+'idk',uq_idk) 
    
'''    
def get_idk_summary(args,pred_idk,misclass_mask,uq):
    tp_idk,mis_idk=0,0
    tp_nidk,mis_nidk=0,0
    
    #ttl_pixel=misclass_mask.shape[0]*misclass_mask.shape[1]
    
    idk_mask=pred_idk.numpy()
    mis_idk = np.count_nonzero(idk_mask*misclass_mask)
    mis_nidk = np.count_nonzero((1-idk_mask)*misclass_mask)
    tp_idk = np.count_nonzero(idk_mask*(1-misclass_mask))
    tp_nidk = np.count_nonzero((1-idk_mask)*(1-misclass_mask))
    
    print('misclass:',np.count_nonzero(misclass_mask),'idk:',np.count_nonzero(idk_mask))
    print('idk-with-miss:',mis_idk,'idk-tp:',tp_idk)
    mis_idk/=np.count_nonzero(misclass_mask)
    mis_nidk/=np.count_nonzero(misclass_mask)
    tp_idk/=np.count_nonzero((1-misclass_mask))
    tp_nidk/=np.count_nonzero((1-misclass_mask))
    
    string=str(tp_idk)+','+str(tp_nidk)+','+str(mis_idk)+','+str(mis_nidk)+'\n'
    
    uq_idk=idk_mask*uq.numpy()
    
    return string,uq_idk
    
def predictions_M1_NOT_IDK(pred_idk, pred_m1, label):
    idk_nmask=pred_idk.numpy()
    idk_nmask=1-idk_nmask
    
    pred_m1= pred_m1.cpu().numpy()
    true_labels = label.cpu().numpy()
    
    pred_m1_tmp=np.where(pred_m1==0,3.0,pred_m1)
    true_labels_tmp=np.where(true_labels==0,3.0,true_labels)
    
    pred_m1_tmp= pred_m1_tmp*idk_nmask
    true_labels_tmp= true_labels_tmp*idk_nmask
    
    pred_m1 = pred_m1_tmp[np.nonzero(pred_m1_tmp)]
    true_labels = true_labels_tmp[np.nonzero(true_labels_tmp)]
    
    a1= np.where(pred_m1==3,0, pred_m1)
    a2= np.where(true_labels==3,0, true_labels)
    
    return torch.as_tensor(a1.copy()), torch.as_tensor(a2.copy())   
'''            

def count_values_for_each_class(num_class,data,pos,Ntest):
    x=np.zeros(num_class)
    for t in range(Ntest):
        c=data[t][pos[0]][pos[1]]
        x[c]+=1
    
    return x

def calculate_uncertainty(predictions,mean_probs, Ntest,num_class=3):
    #for categorical we calculate shanons entropy, higher entropy value means high uncertain
    uncertain_es=np.ones(predictions[0].shape)
    size=predictions[0].shape
    #print('uq shape:',uncertain_es.shape)
    
    for i in range(size[0]):
        for j in range(size[1]):
            
            x=count_values_for_each_class(num_class,predictions,(i,j),Ntest)
            px=x/Ntest
            pxt = np.where(px==0, 0.00001, px)
            et=pxt*np.log(pxt)
            uncertain_es[i][j]=-np.sum(et)
        
            #get the confidence estimate instead entropy
            #uncertain_es[i][j]=torch.max(mean_probs[:,i,j]).item()
        
    return uncertain_es
    
def predict_img(img_file,label_file,model,Ntest,device):
    model.eval()
    img=M1Dataset.preprocess(img_file, False)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    label=torch.as_tensor(M1Dataset.preprocess(label_file, True).copy())
    label = label.to(device=device, dtype=torch.int32)
    all_predictions=[]
    with torch.no_grad():
        output = model(img)          
        mean_probs = F.softmax(output, dim=1)[0]
        all_predictions.append(mean_probs.argmax(dim=0).cpu().numpy())
    
    for t in range(Ntest-1):
        with torch.no_grad():
            output = model(img) 
            probs = F.softmax(output, dim=1)[0]
            mean_probs=torch.add(mean_probs,probs)
            all_predictions.append(probs.argmax(dim=0).cpu().numpy())
    
    mean_probs=torch.div(mean_probs,Ntest)
    uq = calculate_uncertainty(all_predictions,mean_probs,Ntest)
    
    
    return label, mean_probs.argmax(dim=0), torch.as_tensor(uq.copy()), output.numpy()[0]

def set_model(args,device):
    model_m1 = UNet(n_channels=1, n_classes=args.num_class, drop_train=True, drop_rate=args.p, bilinear=True)
    state_dict = torch.load(args.dir_pretrain, map_location=device)
    #model_m1.module.load_state_dict(state_dict) #for running on multi-gpu
    model_m1.load_state_dict(state_dict)
    model_m1.to(device=device)
    
    
    #model_idk=IDK(args.idk_th_min,args.idk_th_max)
    
    return model_m1

    
def m1_idk_predict(args, sample_type, device):
    f1_file=open(args.dir_out+'f1_{}.txt'.format(sample_type),'w')
    #f1nidk_file=open(args.dir_out+'f1_nidk{}.txt'.format(sample_type),'w')
    
    misclass_score=open(args.dir_out+'misclass_m1_{}.txt'.format(sample_type),'w')
    #idk_summary= open(args.dir_out+'idk_score_{}.txt'.format(sample_type),'w')
    #idk_summary.write('file_name,m1-tp-idk,m1-tp-not-idk,m1-mis-idk,m1-mis-not-idk\n')
    
    fids = os.listdir(args.dir_img)
    model_m1 = set_model(args,device)
    
    f1_metric = F1Score(3,'macro')
    name_list=[]
    for i,image in enumerate(fids):
        if image[-3:]!='png':
            continue
        name_list.append(image)
        
        img_file=args.dir_img+image
        label_file=args.dir_mask+image
        true_mask,pred_label_m1, uq, out = predict_img(img_file, 
                                                label_file, model_m1, args.Ntest, device)
        pred_mask=F.one_hot(pred_label_m1, model_m1.n_classes).permute(2, 0, 1).cpu().numpy()
        
        pred_image,_=mask_to_image(args.dir_out+image,pred_mask,args.num_class)
        
        lines,misclass,misclass_mask=plot_misclassification(true_mask.cpu().numpy(), pred_label_m1.cpu().numpy(), [0,1,2])
        print('file:',image)
        
        f1,prec,rec,ttl_f1 = f1_metric(pred_label_m1, true_mask)
        f1_summ=[prec,rec,f1,ttl_f1]
        
        #plot_img_and_mask(args.dir_out+'prediction/'+image,img_file,label_file,pred_image,misclass)
        '''
        final_mis=torch.as_tensor(misclass_mask.copy())
        if sample_type=='test':
            final_mis=torch.ones(pred_label_m1.shape)
        
        pred_idk= model_idk(final_mis, uq)
        idk_summ,uq_idk = get_idk_summary(args,pred_idk,misclass_mask,uq)
        
        
        pred_nidk,label_nidk = predictions_M1_NOT_IDK(pred_idk, pred_label_m1, true_mask)
        f1_nidk,prec_nidk,rec_nidk,ttl_f1_nidk = f1_metric(pred_nidk, label_nidk)
        f1nidk_summ=[prec_nidk,rec_nidk,f1_nidk,ttl_f1_nidk]
        '''
        save_files(image,misclass_score,f1_file, f1_summ, lines)
        save_predictions(args,sample_type, image[:-3],pred_label_m1, 
                         uq,misclass,misclass_mask, out)
        
        
         
        
    f1_file.close() 
    #f1nidk_file.close()
    misclass_score.close() 
    #idk_summary.close()
    
def main():
    
    args=parse_args()

    sample_type='train'
    if args.test_only:
       sample_type='test'
    
    if args.device=='cpu':
        device='cpu'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out+'misclass/')
        os.makedirs(args.dir_out+'uq/')
        os.makedirs(args.dir_out+'prediction/')
        #os.makedirs(args.dir_out+'idk-'+sample_type+'/')
    
    
    m1_idk_predict(args,sample_type,device)
    

        
if __name__ == '__main__':
    main()
    