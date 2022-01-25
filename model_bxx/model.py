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
    
    
    def fit(self, data_train, data_test, tqdm=None,
            lr = 1e-4, lr_lambda=None, weight_decay=1e-4, num_epoch=50, verbose=False, **kwargs):
        '''
        Params:
            - data_train/data_test : Dataloader type.
            - lr_lambda : float, used to decrease the learning rate. 
            - num_epoch : int, default is 50.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if lr_lambda is None:
            lr_lambda = lambda epoch : 0.95 if epoch%10==0 else 1
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
        loss_func = nn.CrossEntropyLoss()
        test_loss_list = []
        train_loss_list = []
        
        # # Separate train data from test data.
        # num_totl = len(df_data)
        # idx_tr = np.random.rand(num_totl) <= 0.8
        # df_train = df_data[idx_tr]
        # df_test = df_data[~idx_tr]
        # num_tr = len(df_train)
        
        it_epoch = range(num_epoch)
        if tqdm is not None:
            it_epoch = tqdm(it_epoch)
        
        for epoch in it_epoch:
            # for j in range(0, num_tr, batch_size):
            #     self.zero_grad()
            #     k = j+batch_size if j+batch_size<num_tr else num_tr
            #     df_tmp = df_train.iloc[j:k]
            #     X = file_loader(df_tmp['itemid'])
            #     y = torch.tensor(df_tmp['hasbird'].to_numpy()).to(self.device).long()
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
                # del X
                # del y
                # print(j)
                
            lr_scheduler.step()
            
            with torch.no_grad():
                train_loss = loss.detach().cpu().numpy()
                train_loss_list.append(train_loss)
                
                _tmp_test_loss = []
                for X, y in data_test:
                    if not isinstance(y, torch.Tensor):
                        y = torch.tensor(y)
                    y = y.to(self.device).long()
                    y_prd = self(X)
                    loss = loss_func(y_prd, y)
                    _tmp_test_loss.append(loss.cpu().numpy())
                
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
