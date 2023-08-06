#region imports
from AlgorithmImports import *
#endregion
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.autograd import Variable
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim, batch_size, num_dir, rnn_module):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.bs = batch_size
        self.nd = num_dir
        if self.nd == 2:
            bidir = True
        else:
            bidir = False
        if rnn_module == "srnn":
            self.rnnmodule = nn.RNN(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=bidir)
        elif rnn_module == "lstm":
            self.rnnmodule = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=bidir)
        self.linear = nn.Linear(self.hidden_dim * self.nd, self.out_dim)

    def forward(self, x):
        out, _ = self.rnnmodule(x)
        out = self.linear(out[-1])
        out = out * 100
        return out


def Windowed_Dataset(series, window_size, stride, batch_size, shuffle):
    """
    params:
        series: time series data
        window_size: K points in this window are used to predict the next (K+1) point
        stride: stride between windows
        batch_size: batch size for training
    return:
        ds_loader: wrap windowed data into pytorch dataloader
    """
    f_s = window_size + 1
    l = len(series)
    ds = torch.from_numpy(series)
    ds = torch.unsqueeze(ds, dim=1)
    ds = [ds[i:i+f_s] for i in range(0, l, stride) if i <= l-f_s]  # each chunk contains k+1 points, the last one is the target
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return ds_loader

def triain():
    window_size = 20
    batch_size = 32
    input_dim = 1
    hidden_dim = 40
    out_dim = 1
    num_layers = 2
    num_dir = 1  # prediction direction, one-sided or two-sided
    num_epochs = 50
    learning_rate = 0.0001
    # rnn_module = "srnn"
    rnn_module = "lstm"

    split_time = int(.9 * len(t))  # 90000
    time_train = t[:split_time]
    x_train = dat[0][:split_time]
    time_test = t[split_time:]
    x_test = dat[0][split_time:]

    train_loader = Windowed_Dataset(x_train, window_size=window_size, stride=1, batch_size=batch_size, shuffle=True)
    test_loader = Windowed_Dataset(x_test, window_size=window_size, stride=1, batch_size=batch_size, shuffle=False)

    model = RNN(input_dim, hidden_dim, num_layers, out_dim, batch_size, num_dir, rnn_module)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    best_loss = 1e+100
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0
        running_test_loss = 0
        for batch_index, item in enumerate(train_loader):
            """
            item shape: batch_size * (window_size+1) * 1
            inputs shape: batch_size * (window_size) * 1
            target(last point in item) shape: batch_size * 1
            """

            inputs = item[0:batch_size, 0:-1]
            inputs = torch.transpose(inputs, 0, 1)
            inputs = inputs.float().to(device)
            target = item[0:batch_size, -1:].squeeze(dim=1)
            target = target.float().to(device)

            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        with torch.no_grad():
            for batch_index, item in enumerate(test_loader):
                inputs_test = item[0:batch_size, 0:-1]
                inputs_test = torch.transpose(inputs_test, 0, 1)
                inputs_test = inputs_test.float().to(device)
                target_test = item[0:batch_size, -1:].squeeze(dim=1)
                target_test = target_test.float().to(device)

                out_test = model(inputs_test)
                loss_test = criterion(out_test, target_test)
                running_test_loss += loss_test.item()
            if epoch % 10 == 0:
                print('Epoch {} : Training loss is {:.4f} \
                    '.format(epoch, running_train_loss / (batch_index + 1)))
                print('Epoch {} : Test loss is {:.4f} \
                    '.format(epoch, running_test_loss / (batch_index + 1)))
            if best_loss > running_test_loss * batch_size / len(x_test):
                torch.save(model, 'Codes/ckpt/ckpt_time_series_{}.pth'.format(rnn_module))
                best_loss = running_test_loss * batch_size / len(x_test)


def prediction(model_name, series, window_size):
    model = torch.load(model_name)
    model.eval()
    model = model.to(device)
    forcast = []

    series_t = torch.tensor(series)
    series_t = torch.unsqueeze(series_t, dim=1)
    for time_step in range(len(series) - window_size):
        Input_time = series_t[time_step:time_step+window_size,:]
        Input_time = Input_time.float().to(device)
        forcast.append(model(Input_time))
    result = forcast #[split_time-window_size:]
    result = [x.detach().cpu().numpy().squeeze() for x in result]
    return result


# prediction
k = 5  # predict dataset k using model trained with dataset 0
srnn_result = prediction('Codes/ckpt/ckpt_time_series_srnn.pth', dat[k], window_size)
# lstm_result = prediction('ckpt_time_series_lstm.pth', series, window_size)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(srnn_result[0:10000], label='predicted')
axes[1].plot(dat[k][window_size:10000+window_size], label='observed')
axes[0].legend()
axes[1].legend()
plt.show(block=True)
