#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:58:24 2022

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
    parser.add_argument('--dir_model', type=str, required= True,
                        help='pretrain model directory')
    parser.add_argument('--dir_data', type=str, default='../data/mechanical_loading_datasets/',
                        help='data directory')
    parser.add_argument('--test_data', type=str, default='500mAh3.xlsx',
                        help='test data filename')
    
    args = parser.parse_args()
    
    return args


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
    
    state_dict = torch.load(args.dir_model, map_location=device)
    model.load_state_dict(state_dict)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = _loss_fn(args)
    
    model.to(device=device)
    return model,optimizer,loss_fn

def set_data(args):
    val_data=DS(args.dir_data, args.history_size, args.time_lag, args.horizon, is_test=True, 
                test_data=args.test_data)
    
    
    val_sampler=None
    val_loader = DataLoader(val_data, batch_size=512, num_workers=0, 
                            pin_memory=True, sampler=val_sampler)
    
    return val_loader

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

def main():
    args=parse_args()
    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model,optimizer,loss_fn = set_model(args,device)
    
    val_loader= set_data(args) 
    
    val_loss, test_forecasts= validate(args, device, model, val_loader, loss_fn)
    print('val loss:',val_loss,test_forecasts.shape)
    
    forecasts_df = pd.DataFrame(test_forecasts, columns=[f'V{i + 1}'
                                    for i in range(args.horizon)])
    forecasts_df.index.name = 'id'
    forecasts_df.to_csv(args.dir_out+'forecast_test.csv')

if __name__ == '__main__':
    main()


