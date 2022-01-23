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
    
    
    def fit(self, dataset=None, tqdm=None,
            lr = 0.1, lr_lambda=None, batch_size=2400, num_epoch=50, verbose=False, **kwargs):
        '''
        Params:
            - dataset : AudioDataset type.
            - lr_lambda : float, used to decrease the learning rate. 
            - batch_size : int, default is 2400.
            - num_epoch : int, default is 50.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
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
            for X, y in dataset:
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y)
                y.to(self.device).long()
                y_prd = self(X)
                loss = loss_func(y_prd, y)
                loss.backward()
                optimizer.step()
                
                # print(j)
                
            lr_scheduler.step()
            
            with torch.no_grad():
                train_loss_list.append(loss.detach().cpu().numpy())
                X = file_loader(df_test['itemid'])
                y = torch.tensor(df_test['hasbird'].to_numpy()).to(self.device).long()
                y_prd = self(X)
                test_loss = loss_func(y_prd, y).cpu().numpy()
                test_loss_list.append(test_loss)
            it_epoch.set_postfix({'test loss': test_loss, 'train loss': loss.cpu().detach().numpy()})
                
        fig, axs = plt.subplots(2)
        axs[0].plot(np.arange(1, num_epoch+1), test_loss_list, color='r', label='test loss')
        axs[0].set_xlim(left=1, right=num_epoch+2)
        axs[0].legend()
        axs[0].grid()
        axs[1].plot(np.arange(1, num_epoch+1), train_loss_list, color='b', label='train loss')
        axs[1].set_xlim(left=1, right=num_epoch+2)
        axs[1].legend()
        axs[1].grid()
        plt.show()
