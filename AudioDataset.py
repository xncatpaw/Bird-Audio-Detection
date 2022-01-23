import os
import shutil
import math
from itertools import accumulate
from collections import Counter
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

class AudioDataset(Dataset):
    
    def __init__(self, label_df, audio_dir, transforms, target_sample_rate, num_sample, device, is_train_or_valid=True):
        super().__init__()
        self.label_df = label_df
        self.audio_dir = audio_dir
        self.device = device
        if isinstance(transforms, list):
            self.transforms = [transform.to(self.device) for transform in transforms]
        else:
            self.transforms = transforms.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_sample = num_sample
        self.is_train_or_valid = is_train_or_valid
        
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, index):
        # no indexOutOfRange exception
        audio_sample_path = self._get_sample_path(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        if signal.dim() == 1:
            signal = signal[None,:]
        signal = signal.to(self.device)
        signal = self._resample(signal, sample_rate)
        signal = self._mix(signal)
        signal = self._truncate(signal)
        signal = self._pad(signal)
        if isinstance(self.transforms, list):
            for transform in self.transforms:
                signal = transform(signal)
        else:
            signal = self.transforms(signal)
        
        if self.is_train_or_valid:
            audio_label = self._get_sample_label(index)
            return signal, audio_label
        else:
            return signal
    
    def _resample(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
        
    def _truncate(self, signal):
        if signal.shape[1] > self.num_sample:
            start = (signal.shape[1] - self.num_sample) // 2
            signal = signal[:, start: start + self.num_sample]
        return signal
    
    def _pad(self,signal):
        if signal.shape[1] < self.num_sample:
            num_pad = self.num_sample - signal.shape[1]
            left_pad = num_pad // 2
            signal = F.pad(signal, (left_pad, num_pad - left_pad))
        return signal
    
    def _get_sample_path(self, index):
        filename = str(self.label_df.iloc[index, 0])
        path = os.path.join(self.audio_dir, filename)
        return path
    
    def _get_sample_label(self, index):
        return self.label_df.iloc[index, 1]