#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:31:59 2021

@author: oit
"""

import torch
import numpy as np
from torch import Tensor

class F1Score:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self,num_class,average='macro'):
        self.average = average
        self.classes=num_class
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')
        
    #@staticmethod   
    def calc_f1_count_for_label(self,predictions,labels, label_id):
        # label count
        true_count = torch.eq(labels, label_id).sum()
        if true_count==0:
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), torch.tensor(0)
        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(true_positive),
                                precision)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        #print('pre, rec, f1:',precision.item(),recall.item(),f1.item())
        return f1, precision, recall, true_count

    def __call__(self, predictions, labels):
        pred1=predictions.flatten()
        labels1=labels.flatten()
        num_label=self.classes#len(labels1.unique())
        f1_score = 0
        f1_arr=np.zeros(num_label)
        #true_arr=np.zeros(num_label)
        prec_arr=np.zeros(num_label)
        rec_arr=np.zeros(num_label)
        for label_id in range(0, num_label):
            f1, prec, rec, true_count = self.calc_f1_count_for_label(pred1, labels1, label_id)
            f1_arr[label_id]=f1.item()
            #true_arr[label_id]=true_count.item()
            prec_arr[label_id]=prec.item()
            rec_arr[label_id]=rec.item()
            if self.average == 'weighted':
                f1_score += f1 * true_count
            elif self.average == 'macro':
                f1_score += f1.item()
            
            
        if self.average == 'weighted':
            f1_score = torch.div(f1_score, len(labels1))
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, num_label)

        return f1_arr,prec_arr,rec_arr,f1_score.item()

