import os
import time
from itertools import accumulate
from IPython import display
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return list(accumulate(self.times))
    
class Custom_Plot:
    def __init__(self, xlim=[0,10], xlabel='epoch', legends=['loss', 'train_acc', 'test_acc'], fmts=('-', 'm--', 'g-.', 'r:')):
        self.X = None
        self.Y = None
        self.xlim = xlim
        self.xlabel = xlabel
        self.legends = legends
        self.fmts = fmts
        
    def reinit(self):
        self.X = []
        self.Y = []
        
    def add(self, x, y):
        if not hasattr(y, '__len__'):
            y = [y]
        n = len(y)
        if not hasattr(x, '__len__'):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
    
    def plot(self, save=False, name='fig', path=None):
        plt.clf()
        display.clear_output(wait=True)
        fig, ax = plt.subplots(1, 1, figsize=(5,5))
        for x, y, fmt, legend in zip(self.X, self.Y, self.fmts, self.legends):
            ax.plot(x, y, fmt, label=legend)
        ax.set_xlabel(self.xlabel)
        ax.set_xlim(*self.xlim)
        ax.legend(loc='best')
        plt.show()
        
        if save:
            if not path:
                path = os.cwd()
            os.makedirs(path, exist_ok=True)
            fig.savefig(os.path.join(path, name), format='svg')

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, device,
          num_epochs=10, lr=1e-4, wd=1e-4, lr_period=2, lr_decay=0.9,
          model_name='nn',
          save_param=True, param_path=None,
          plot=True, save_fig=True, fig_path=None,
          initialize = False):
    
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    if initialize:
        net.apply(init_weights)

    print('training on', device)
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    
    loss = nn.CrossEntropyLoss(reduction='none')
    plotter = Custom_Plot(xlim=[0, num_epochs], xlabel='epoch', legends=['train_loss', 'train_acc', 'test_acc'])
    timer, num_batches = Timer(), len(train_iter)
    curr_time = datetime.now().strftime('%Y-%b-%d-%Hh%m')
    
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l, accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if ((i + 1) % (num_batches // 5) == 0 or i == num_batches - 1):
                plotter.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
                if plot:
                    plotter.plot()
                else:
                    print(f'epoch {epoch}, batch {i+1}, train_loss: {train_l:.3f}, train_accuracy: {train_acc:.3f}')
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        plotter.add(epoch + 1, (None, None, test_acc))
        if save_fig and epoch == num_epochs - 1:
            filename = model_name + '_' + curr_time + '.svg'
            plotter.plot(save=True, name=filename, path=fig_path)
        elif plot:
            plotter.plot()
        print(f'epoch {epoch} trained, train_loss: {train_l:.3f}, train_accuracy: {train_acc:.3f}, test_accuracy: {test_acc:.3f}')
        scheduler.step()
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    
    if save_param:
        if not param_path:
            param_path = os.cwd()
        os.makedirs(param_path, exist_ok=True)
        filename = model_name + '_params_' + curr_time
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': net.state_dict(),
            'loss': loss,
        },
            os.path.join(param_path, filename))