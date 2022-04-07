#!/usr/bin/env python3
"""
Created on Tue Mar  8 07:05:24 2022
@author: oit
"""
import os
import numpy as np
import pandas as pd
import torch

class DataSampler:
    def __init__(self, data_dir, in_size, time_lag, out_size, is_test, test_data=None):
        super().__init__()
        ids=[]
        self.labels=[0.2, 0.4, 0.6, 0.8, 1.0]
        self.insample_size = in_size
        self.time_lag = time_lag
        self.horizon = out_size
        self.ts_len = in_size+time_lag+out_size
        self.cutpoints=[]
        #self.forecast_sidx=[]
        self.ts_index=[]
        self.timeseries=[]
        num_ts=0
        if is_test:
           data = pd.read_excel(data_dir+test_data, engine='openpyxl', 
                                sheet_name= 'temperature', 
                                header=[0,1])
           ts = data[(1.0, 'temperature/℃')].to_numpy()
           ts = ts[~np.isnan(ts)]
           ts1 = (ts - np.mean(ts))/np.var(ts)
           
           self.timeseries.append(ts1)
           forecast_sidx = len(ts)-self.horizon
           cp = np.arange(self.insample_size, forecast_sidx-self.time_lag,1, dtype= int)
           total_cp = len(cp) 
           self.ts_index+= [num_ts]*total_cp
           self.cutpoints+=list(cp)
           
           '''
           index = 0
           while index< len(ts): 
               if index+self.ts_len< len(ts):
                   
                   forecast_eidx = min(index+3*horizon, len(ts)) 
                   cp = np.arange(index+horizon, forecast_eidx-horizon, 1, dtype= int)
                   total_cp = len(cp) 
                   self.ts_index+= [num_ts]*total_cp
                   self.cutpoints+=list(cp)
                   self.forecast_sidx+=[index+2*horizon]*total_cp
                   
                   
                  # print('index:',cp[0]-self.insample_size,
                   #      cp[0], cp[-1], cp[-1]+1, forecast_idx)
               index+= self.ts_len
            '''  
                     
        else:
            fids=os.listdir(data_dir)
        
            for file in fids:
                if file[-4:]!='xlsx':
                    continue
                ids.append(file)
        
            for fid in ids:
                data = pd.read_excel(data_dir+file, engine='openpyxl', 
                                     sheet_name= 'temperature', header=[0,1])
                for l in self.labels:
                    if fid==test_data and l==1.0:
                        continue
                    ts = data[(l, 'temperature/℃')].to_numpy()
                    ts = ts[~np.isnan(ts)]
                    ts1 = (ts - np.mean(ts))/np.var(ts)
                    self.timeseries.append(ts1)
                    
                    forecast_sidx = len(ts)-self.horizon
                    cp = np.arange(self.insample_size, forecast_sidx-self.time_lag, 1, 
                                   dtype= int)
                    total_cp = len(cp) 
                    self.ts_index+= [num_ts]*total_cp
                    self.cutpoints+=list(cp)
                    '''
                    index = 0
                    while index< len(ts):
                        if index+self.ts_len< len(ts):
                            forecast_idx = min(index+3*horizon, len(ts)) 
                            cp = np.arange(index+horizon, forecast_idx-horizon, 
                                           1, dtype= int)
                            total_cp = len(cp) 
                            self.ts_index+= [num_ts]*total_cp
                            self.forecast_sidx+=[index+2*horizon]*total_cp
                            self.cutpoints+=list(cp)      
                        
                        index+= self.ts_len
                    '''   
                    num_ts+=1
        
        print(len(self.cutpoints),len(self.ts_index))

    def __len__(self): 
        return len(self.cutpoints)
    
    def __getitem__(self,i):
        
        start_id = self.cutpoints[i]-self.insample_size
        sid_y = self.cutpoints[i]+self.time_lag
        eid = sid_y+self.horizon
        ts = self.timeseries[self.ts_index[i]]
         
        X= ts[start_id:self.cutpoints[i]]
        y = ts[sid_y:eid]
        
        return torch.tensor(X), torch.tensor(y)

'''
def main():
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    
    data_dir = '../../data/mechanical_loading_datasets/'
    #train_dataset=DataSampler(data_dir, 50, 50, 10, is_test=True, test_data= '500mAh3.xlsx')
    train_dataset=DataSampler(data_dir, 50, 50, 10, is_test=False)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, 
                              shuffle= False,
                              num_workers=0, pin_memory=True, 
                              sampler=None)
    print('data size:',len(train_loader))
    
    for idx, (X, y) in enumerate(train_loader):
        #print('idx:',idx, X.shape, y.shape)
        xn = torch.nonzero(torch.isnan(X.view(-1)))
        yn = torch.nonzero(torch.isnan(y.view(-1)))
        print('idx:', idx, xn.shape[0], yn.shape[0])
        #if idx>10:
         #   break
    
    
if __name__ == '__main__':
    main() 
'''
