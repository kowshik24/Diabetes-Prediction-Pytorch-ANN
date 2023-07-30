import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Creating Model using pytorch
class ANN_Model(nn.Module):
    def __init__(self,input_features=8,hidden1=256,hidden2=64,hidden3=64,out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features,hidden1)
        self.f_connected2 = nn.Linear(hidden1,hidden2)
        self.f_connected3 = nn.Linear(hidden2,hidden3)
        self.out = nn.Linear(hidden2,out_features)
    def forward(self,x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = F.relu(self.f_connected3(x))
        x = self.out(x)
        return x