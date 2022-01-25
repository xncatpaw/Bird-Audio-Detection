'''
The Resnet structure model.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import sys
sys.path.insert(0, '../')

from .model import Model
from model_utils import _resnet_block


class ModelResNet(Model):
    def __init__(self, CUDA=False):
        super(ModelResNet, self).__init__(CUDA)
        
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*_resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*_resnet_block(64, 128, 2))
        b4 = nn.Sequential(*_resnet_block(128, 256, 2))
        b5 = nn.Sequential(*_resnet_block(256, 512, 2))
        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1,1)),
                            nn.Flatten(), nn.Linear(512, 2))
        
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
        self.net.apply(init_weights)
        self.to(self.device)
        
        
    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).to(self.device)
        return self.net(X)
