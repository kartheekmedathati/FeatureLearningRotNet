from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import os
import torchnet as tnt
import utils
import PIL
import pickle
from tqdm import tqdm
import time

from . import Algorithm
from pdb import set_trace as breakpoint

#from random import seed
#from random import randint
# seed random number generator
np.random.seed(42)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class ClassificationModel(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()
        self.tensors['labels1'] = torch.LongTensor()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        start = time.time()
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        self.tensors['labels1'].resize_(batch[2].size()).copy_(batch[2])
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        labels1 = self.tensors['labels1']
        batch_load_time = time.time() - start
        #print("Labels:", labels)
        #print("HLabels:", labels1)
        #********************************************************

        #********************************************************
        start = time.time()
        if do_train: # zero the gradients
            self.optimizers['model'].zero_grad()
        #********************************************************

        #***************** SET TORCH VARIABLES ******************
        dataX_var = torch.autograd.Variable(dataX, volatile=(not do_train))
        labels_var = torch.autograd.Variable(labels, requires_grad=False)
        labels_var1 = torch.autograd.Variable(labels1, requires_grad=False)
        #print("Getting called Label size:",labels_var.size())
        #********************************************************

        #************ FORWARD THROUGH NET ***********************
        fc_feat, pred_var, pred_var1 = self.networks['model'](dataX_var)
        #pred_var = self.networks['model'](dataX_var)
        #print("Shape of predictions:",pred_var.size())
        #********************************************************
        
        #*************** COMPUTE LOSSES *************************
        record = {}
        loss_mse = self.criterions['mseloss'](fc_feat[0::16,:],fc_feat[np.random.randint(0,16)::16,:])
        #loss_mse = self.criterions['mseloss'](fc_feat[0::16,:],fc_feat[np.random.randint(0,16)::16,:])
        loss_cross_entropy1 = self.criterions['loss'](pred_var, labels_var) 
        loss_cross_entropy2 = self.criterions['loss'](pred_var1,labels_var1)
        print("Loss cross entroy 1:", loss_cross_entropy1.data, "cross entropy2:",loss_cross_entropy2.data, "MSE:", loss_mse.data)
        loss_total = loss_cross_entropy1 + loss_cross_entropy2 + loss_mse
        #loss_total = 0.05*loss_cross_entropy1 + 0.05*loss_cross_entropy2 + 0.9*loss_mse
        #loss_total = self.criterions['loss'](pred_var, labels_var) + self.criterions['loss'](pred_var1,labels_var1) +self.criterions['mseloss'](fc_feat[0::16,:],fc_feat[np.random.randint(0,16)::16,:])
        #loss_total = self.criterions['loss'](pred_var, labels_var) 
        #print("Prediction shape:", pred_var.size())
        record['prec1'] = accuracy(pred_var.data, labels, topk=(1,))[0]#[0]
        record['loss'] = loss_total.data#[0]
        #********************************************************

        #****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_total.backward()
            self.optimizers['model'].step()
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)
        return record
