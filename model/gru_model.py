from torch import nn
import pandas as pd
import numpy as np
import random
import torch

pd.options.mode.chained_assignment = None
np.random.seed(1)
random.seed(2)
torch.random.manual_seed(3)


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, num_layers=1, drop_prob=0):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True,
                          dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out.flatten(), h

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().requires_grad_().to(device)
        nn.init.kaiming_normal_(hidden, a=0, mode='fan_out')
        return hidden
