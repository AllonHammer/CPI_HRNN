import random
import pandas as pd
import numpy as np
import torch
import torch.functional as F
from torch import nn
pd.options.mode.chained_assignment = None
np.random.seed(1)
random.seed(2)
torch.random.manual_seed(3)

class CustomLoss(nn.Module):
    def __init__(self, parent_weights=None, son_weights=None, n=None):
        super(CustomLoss, self).__init__()
        self.parent_weights = parent_weights
        self.son_weights = son_weights
        self.n = n

    def forward(self, y_true, y_pred, son_weights):
        self.son_weights = son_weights
        if self.parent_weights is None or self.son_weights is None:
            self.n = 0
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        mse = torch.mean((y_pred - y_true)**2)  # calcaulte the  mean squared error
        norm = torch.sum((self.parent_weights - self.son_weights)**2)  # calculate sum of all indices in norm
        return self.n * norm + mse
