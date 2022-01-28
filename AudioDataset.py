import os
from itertools import accumulate
from collections import Counter
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import time
class AudioDataset(Dataset):
    
    def __init__(self, label_df, audio_dir, transforms, target_sample_rate, length, device, is_train_or_valid=True, do_augmentation = False):
        super().__init__()
        self.label_df = label_df
        self.audio_dir = audio_dir
        self.device = device
        self.do_augmentation = do_augmentation
        pitch_shiftor = torchaudio.transforms.PitchShift(
            target_sample_rate, torch.randint(low=0,high=6,size=(1,))
        )

        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=1024,
            hop_length=512
        )

        freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=30)

        melscale = torchaudio.transforms.MelScale(
            sample_rate=target_sample_rate,
            n_mels=64,
            n_stft=513
        )
        if self.do_augmentation:
            transforms = [spectrogram, freq_masking, melscale]
        else:
            transforms = [spectrogram, melscale]
        self.transforms = transforms
        if isinstance(transforms, list):
            self.transforms = [transform.to(self.device) for transform in transforms]
        else:
            self.transforms = transforms.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.length = length
        self.num_sample = self.target_sample_rate * self.length
        self.is_train_or_valid = is_train_or_valid
        
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, index):
        # no indexOutOfRange exception
        #A = time.time()
        audio_sample_path = self._get_sample_path(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        
        if self.do_augmentation:
            signal = signal + (torch.randn(signal.shape)-0.5)*0.002
        if signal.dim() == 1:
            signal = signal[None,:]

        signal = signal.to(self.device)
        
        signal = self._resample(signal, sample_rate)
        signal = self._mix(signal)
        signal = self._truncate(signal)
        signal = self._pad(signal)

        #B = time.time()

        if isinstance(self.transforms, list):
            for transform in self.transforms:
                signal = transform(signal)
        else:
            signal = self.transforms(signal)
        
        #C = time.time()
        
        #signal = signal.to(self.device)

        #D = time.time()
        #print(B-A,C-B,D-C)
        #print(signal.shape)
        
        if self.is_train_or_valid:
            audio_label = self._get_sample_label(index)
            return signal, audio_label
        else:
            return signal
    
    def _resample(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate).to(self.device)
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