import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from AudioDataset import AudioDataset
import utils, advanced_resnet_model
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

if __name__ == "__main__":
    
    # hyperparams
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(os.path.dirname(project_dir), 'data') # '../../data'

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    sample_rate = 16000
    length = 10
    mel_sepectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    batch_size = 32 # to be changed wrt gpu memory
    num_worker = 2 # to be changed wrt gpu memory

    num_epochs = 4
    lr = 1e-4
    wd = 1e-4
    lr_period = 2
    lr_decay = 0.9


    # dataloader
    train_df = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'), sep=',')
    valid_df = pd.read_csv(os.path.join(data_dir, 'valid_labels.csv'), sep=',')
    train_dataset = AudioDataset(train_df, os.path.join(data_dir, 'train'), mel_sepectrogram, sample_rate, length, device, is_train_or_valid=True)
    valid_dataset = AudioDataset(valid_df, os.path.join(data_dir, 'valid'), mel_sepectrogram, sample_rate, length, device, is_train_or_valid=True)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_worker)
    valid_iter = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)


    # model
    net = advanced_resnet_model.wide_resnet50_2()

    # train
    utils.train(net, train_iter, valid_iter, device,
                num_epochs=num_epochs, lr_decay=lr_decay, wd=wd, lr_period=lr_period,
                model_name='resnet18',
                save_param=True, param_path='./model_params',
                plot=False, save_fig=True, fig_path='./model_figs',
                initialize=True)


