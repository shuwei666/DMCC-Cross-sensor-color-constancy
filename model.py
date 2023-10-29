"""
 Constructs network architecture
 If you use this code, please cite the following paper:
 Shuwei Yue and Minchen Wei. "Effective cross-sensor color constancy using a dual-mapping strategy" in JOSA A 2023.

"""
__author__ = "Shuwei Yue"
__credits__ = ["Shuwei Yue"]

import torch.nn as nn
from torch.nn.init import kaiming_uniform_
import torch.nn.functional as F
import torch


class Dmcc(nn.Module):
    def __init__(self, in_features=8, neurons=11, out_features=2,
                 hidden_layer_num=5, l1_weight=0.00001):
        """
        The DMCC net, i.e., Simple MLP net based on 11 neurons in 5 hidden layers, with 2 linear layers,
         i.e., the first_layer and last_layer
        """
        super(Dmcc, self).__init__()
        self.first_layer = nn.Linear(in_features, neurons)
        kaiming_uniform_(self.first_layer.weight, nonlinearity='relu')
        self.hidden_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(neurons, neurons),
                nn.ReLU(inplace=True),
            )
            for _ in range(hidden_layer_num)
        ])
        for layer in self.hidden_layer:
            kaiming_uniform_(layer[0].weight, nonlinearity='relu')
        self.last_layer = nn.Linear(neurons, out_features)
        kaiming_uniform_(self.last_layer.weight, nonlinearity='relu')
        self.l1_weight = l1_weight

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.first_layer(x))
        for layer in self.hidden_layer:
            x = layer(x)

        out = self.last_layer(x)

        l1_loss = 0
        for name, param in self.named_parameters():
            if 'bias' not in name:
                l1_loss += torch.norm(param, p=1)
        l1_loss *= self.l1_weight

        return out, l1_loss





