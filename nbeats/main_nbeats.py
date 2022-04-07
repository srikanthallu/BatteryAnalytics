#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 20:34:46 2022

@author: oit
"""

import sys
import os
import argparse
from datetime import datetime
import time
from time import sleep
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.datasampler import DataSampler as DS
from torch.utils.data import DataLoader
from utils import generic, interpretable
#import metrics as me
from pytorch_forecasting.metrics import MAPE, MASE, RMSE, SMAPE

def parse_args():
    parser = argparse.ArgumentParser('argument for training M1')
    parser.add_argument('-n','--nodes', default=1,type=int, metavar='N')
    parser.add_argument('--gpus', default=2, type=int, help='number of gpus')
    parser.add_argument('--device', default='cpu', type=str, help='gpu/cpu')
    parser.add_argument('--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='repeat data')
    parser.add_argument('--repeat', default=10, type=int, help='repeat data')
    parser.add_argument('--lookback', default=7, type=int, #nargs='+', 
                        help='backcast window')
    parser.add_argument('-hs','--history_size', default=50, type=int, metavar='N',
                        help='input sequence length')
    parser.add_argument('--horizon', default=10, type=int,
                        help='forecast horizon length')
    parser.add_argument('--time_lag', default=50, type=int,
                        help='time lag from input start seq to forecast start time')
    parser.add_argument('--epochs', default=22000, type=int, 
                        help='number iterations')
    parser.add_argument('--layer_size', default=512, type=int,
                        help='layer size')
    parser.add_argument('--layers', default=4, type=int, help='number layers')
    parser.add_argument('--stacks', default=50, type=int, help='number stacks')
    parser.add_argument('--degree', default=3, type=int, help='degree of polynomial')
    parser.add_argument('--harmonic', default=2, type=int, help='number harmonics')
    parser.add_argument('--loss', default='MAPE', type=str, help='loss type')
    parser.add_argument('--model_type', default='generic', type=str, 
                        help='basis expansion type')
    parser.add_argument('--save_freq', default=100, type=int, 
                        help='save model after x epochs')
    parser.add_argument('--dir_out', type=str, required= True,
                        help='output directory')
    parser.add_argument('--dir_data', type=str, default='../data/mechanical_loading_datasets/',
                        help='data directory')
    parser.add_argument('--test_data', type=str, default='500mAh3.xlsx',
                        help='test data filename')
    
    args = parser.parse_args()
    
    return args
'''
def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

def mape_loss(forecast, target, mask):
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    weights = divide_no_nan(mask, target)
    return t.mean(t.abs((forecast - target) * weights))

'''
def _loss_fn(args):
    if args.loss == 'MAPE':
        criterion= MAPE()
    elif args.loss == 'MASE':
        criterion= MASE()
    elif args.loss == 'SMAPE':
        criterion= SMAPE()
    elif args.loss == 'RMSE':
        criterion= RMSE()
    
    return criterion

def set_model(args,device):
    input_size= args.lookback*args.horizon

    if args.model_type == 'interpretable':
        model = interpretable(input_size=args.history_size, 
                              output_size=args.horizon, trend_blocks= args.stacks,
                              trend_layers= args.layers, 
                              trend_layer_size = args.layer_size, 
                              degree_of_polynomial= args.degree,
                              seasonality_blocks = args.stacks,
                              seasonality_layers = args.layers,
                              seasonality_layer_size = args.layer_size,
                              num_of_harmonics = args.harmonic)
    
    elif args.model_type == 'generic':
        model = generic(input_size=args.history_size, 
                        output_size=args.horizon,
                        stacks=args.stacks, layers= args.layers, layer_size=args.layer_size)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = _loss_fn(args)
    lr_decay_step = args.epochs // 3
    if lr_decay_step == 0:
        lr_decay_step = 1
    
    model.to(device=device)
    return model,optimizer,lr_decay_step,loss_fn

def set_data(args, rank):
    train_data=DS(args.dir_data, args.history_size, args.time_lag, args.horizon, 
                  is_test=False)
    #train_sampler = DistributedSampler(train_data,num_replicas=args.world_size, 
     #                                  rank=rank, shuffle=False)
    train_sampler=None
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=8, 
                              pin_memory=True, sampler=train_sampler)
    
    val_data=DS(args.dir_data, args.history_size, args.time_lag, args.horizon, is_test=True, 
                test_data=args.test_data)
    
    #val_sampler = DistributedSampler(val_data,num_replicas=args.world_size, 
     #                                  rank=rank, shuffle=False)
    val_sampler=None
    val_loader = DataLoader(val_data, batch_size=512, num_workers=0, 
                            pin_memory=True, sampler=val_sampler)
    
    return train_loader, val_loader

def validate(args, device, model, val_loader, loss_fn):
    val_losses=[]
    forecasts = []
    model.eval()
    print('val loader:',len(val_loader)) 
    with torch.no_grad():
        for idx, (X, y) in enumerate(val_loader):
            #if torch.cuda.is_available():
                #X = X.cuda(non_blocking=True)
                #y = y.cuda(non_blocking=True)
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            
            window_forecast = model(X)
            val_loss=loss_fn(window_forecast, y)
            val_losses.append(val_loss.item())
            
            window_forecast = window_forecast.cpu().detach().numpy()
            
            
            if len(forecasts) == 0:
                forecasts = window_forecast  
            else:
                forecasts=np.concatenate([forecasts, window_forecast], axis=0)
    
    return np.average(val_losses),forecasts

def train(device,args):
    '''
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    '''
    model,optimizer,lr_decay_step,loss_fn = set_model(args,device)
    
    train_loader, val_loader= set_data(args,rank=3) 
    
    num_batches=len(train_loader)*args.batch_size
    
    
    loss_file=open(args.dir_out+'loss.txt','w')
    loss_file.write('epoch,loss,val_loss\n')
    start = time.time()
    
    for epoch in range(1, args.epochs+1):
        epoch_loss=0
        model.train()
        for idx, (X, y) in enumerate(train_loader):
            #if torch.cuda.is_available():
                #X = X.cuda(non_blocking=True)
                #y = y.cuda(non_blocking=True)
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
                
            optimizer.zero_grad()
            forecast = model(X)
            #print('forecast:',forecast)
            training_loss = loss_fn(forecast, y)
                
            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += training_loss.item()
            BT = time.time()-start
            if (idx + 1) % 20 == 0:
                print('Train: [{}][{}/{}]\t'
                  'BT {:.3f}\t'
                  'loss {:.4f}'.format(
                   epoch, idx + 1, len(train_loader), BT,
                   epoch_loss/num_batches))
            sys.stdout.flush()
                
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate * 0.5 ** (epoch // lr_decay_step)
        
            
        #if torch.distributed.get_rank() == 0:
        epoch_loss/=num_batches
                
        string1=str(epoch)+','+str(epoch_loss)+'\n'
        loss_file.write(string1)
        if (epoch)%args.save_freq == 0:
            model_save_name=str(args.dir_out+'ckpt_epoch{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_save_name)
        
    val_loss, test_forecasts= validate(args, device, model, val_loader, loss_fn)
    print('val loss:',val_loss,test_forecasts.shape)
    #if gpu == 0:
    print("Training complete in: " + str(time.time() - start))
    model_save_name=str(args.dir_out+'last_ckpt.pth'.format(epoch))
    torch.save(model.state_dict(), model_save_name)
    forecasts_df = pd.DataFrame(test_forecasts, columns=[f'V{i + 1}'
                                    for i in range(args.horizon)])
    forecasts_df.index.name = 'id'
    forecasts_df.to_csv(args.dir_out+'forecast_test.csv')
    loss_file.close()
    
def main():
    args=parse_args()
    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)
    '''
    args.world_size = args.gpus * args.nodes             
    os.environ['MASTER_ADDR'] = 'localhost'              
    os.environ['MASTER_PORT'] = '12355'                      
    mp.spawn(train, nprocs=args.gpus,args=(args,)) 
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device,args)


if __name__ == '__main__':
    main()
