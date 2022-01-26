'''
Here to define the deep model template.
'''
import abc

import numpy as np
import pandas as pd
from scipy.io import wavfile
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module, abc.ABC):
    '''
    Base class of model.
    '''
    def __init__(self, CUDA=False):
        nn.Module.__init__(self)
        if CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
    
    
    @abc.abstractmethod
    def forward(self, X):
        pass
    
    
    def predict(self, X):
        with torch.no_grad():
            u = self(X)
        return u.cpu().numpy()
    
    
    def fit(self, data_train, data_test, data_unlabel=None, tqdm=None,
            lr = 1e-4, lr_lambda=None, weight_decay=1e-4, num_epoch=50, verbose=False, **kwargs):
        '''
        Params:
            - data_train/data_test : Dataloader type.
            - lr_lambda : float, used to decrease the learning rate. 
            - num_epoch : int, default is 50.
            - pseudo : bool, default is False. Indicating whether use the pseudo-label method.
            - n_batch_train : int, default is 40. The number of pseudo-training batches before one epoch of real data training. 
            - alpha_weight : func : int -> float. Indicating pseudo-training weight at eath step.
        '''
        # Parse args.
        pseudo = False
        if 'pseudo' in kwargs and kwargs['pseudo']:
            pseudo = True
            assert data_unlabel is not None
            N_BATCH_TRAIN = kwargs['n_batch_train'] if 'n_batch_train' in kwargs else 40
            
            if 'alpha_weight' in kwargs:
                alpha_weight = kwargs['alpha_weight']
            else:
                T2 = num_epoch / 2 * len(data_unlabel) / N_BATCH_TRAIN
                def alpha_weight(step):
                    if step <= T2:
                        return step/T2 * 0.5
                    else:
                        return 0.5
            
            
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if lr_lambda is None:
            lr_lambda = lambda epoch : 0.95 if epoch%10==0 else 1
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
        loss_func = nn.CrossEntropyLoss()
        test_loss_list = []
        train_loss_list = []
        
        
        it_epoch = range(num_epoch)
        if tqdm is not None:
            it_epoch = tqdm(it_epoch)
        
        for epoch in it_epoch:
            _tmp_lst_tr = []
            if not pseudo: # In case where the pseudo label is not used.
                self.train()
                it_train = data_train
                if tqdm is not None:
                    it_train = tqdm(it_train, leave=False)
                for X, y in it_train:
                    optimizer.zero_grad()
                    if not isinstance(y, torch.Tensor):
                        y = torch.tensor(y)
                    y = y.to(self.device).long()
                    y_prd = self(X)
                    loss = loss_func(y_prd, y)
                    loss.backward()
                    optimizer.step()

                    train_loss = loss.detach().cpu().numpy()
                    it_train.set_postfix({'crt train loss': train_loss})
                    _tmp_lst_tr.append(train_loss)
                    
            else: # In case where we use the pseudo label.
                step = 1
                it_unlabel = enumerate(data_unlabel)
                if tqdm is not None: 
                    it_unlabel = tqdm(it_unlabel, leave=False, total=len(data_unlabel))
                for batch_idx, X_unlabel in it_unlabel:
                    self.eval()
                    y_unlabel = self(X_unlabel)
                    _, pseudo_label = torch.max(y_unlabel, 1)
                    del y_unlabel
                    
                    self.train()   
                    y_prd = self(X_unlabel)
                    unlabel_loss = alpha_weight(step) * loss_func(y_prd, pseudo_label)
                    optimizer.zero_grad()
                    unlabel_loss.backward()
                    optimizer.step()
                    crt_ulbl_loss = unlabel_loss.detach().cpu().numpy()/alpha_weight(step)
                    it_unlabel.set_postfix({'crt unlabel loss': crt_ulbl_loss})
                    
                    del X_unlabel
                    del pseudo_label
                    del y_prd
                    
                    if batch_idx % N_BATCH_TRAIN == 0: # Train with the labeled data.
                        it_train = data_train
                        if tqdm is not None:
                            it_train = tqdm(it_train, leave=False)
                        for X, y in it_train:
                            optimizer.zero_grad()
                            if not isinstance(y, torch.Tensor):
                                y = torch.tensor(y)
                            y = y.to(self.device).long()
                            y_prd = self(X)
                            loss = loss_func(y_prd, y)
                            loss.backward()
                            optimizer.step()

                            train_loss = loss.detach().cpu().numpy()
                            it_train.set_postfix({'crt train loss': train_loss})
                            _tmp_lst_tr.append(train_loss)
                            del X
                            del y
                            del y_prd

                        # Now we increment step by 1
                        step += 1
                
                
            lr_scheduler.step()
            
            self.eval()
            with torch.no_grad():
                train_loss = np.mean(_tmp_lst_tr)
                train_loss_list.append(train_loss)
                
                _tmp_test_loss = []
                for X, y in data_test:
                    if not isinstance(y, torch.Tensor):
                        y = torch.tensor(y)
                    y = y.to(self.device).long()
                    y_prd = self(X)
                    loss = loss_func(y_prd, y)
                    _tmp_test_loss.append(loss.cpu().numpy())
                    del X
                    del y
                    del y_prd
                
                test_loss = np.mean(_tmp_test_loss)
                test_loss_list.append(test_loss)
            it_epoch.set_postfix({'test loss': test_loss, 'train loss': train_loss})
                
        fig, axs = plt.subplots()
        axs.plot(np.arange(1, num_epoch+1), test_loss_list, color='r', label='test loss')
        axs.plot(np.arange(1, num_epoch+1), train_loss_list, color='b', label='train loss')
        axs.set_xlim(left=1, right=num_epoch+2)
        axs.legend()
        axs.grid()
        plt.show()
