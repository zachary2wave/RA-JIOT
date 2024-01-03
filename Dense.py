import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class DenseNet(nn.Module):
    def __init__(self, APn, UEn, hidden_layer=[2048, 2048, 2048, 2048]):
        super(DenseNet, self).__init__()
        self.APn = APn
        self.UEn = UEn

        self.L1 = nn.Linear((UEn+1)*APn, hidden_layer[0], bias=True)
        self.L2 = nn.Linear(hidden_layer[0], hidden_layer[1], bias=True)
        self.L3 = nn.Linear(hidden_layer[1], hidden_layer[2], bias=True)
        self.L4 = nn.Linear(hidden_layer[2], hidden_layer[3], bias=True)
        self.L5 = nn.Linear(hidden_layer[3], UEn*APn, bias=True)
        self.L6 = nn.Linear(hidden_layer[3], APn, bias=True)


    def forward(self, x):
        x = x.flatten().squeeze(0)
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        x = F.relu(self.L4(x))

        UAoutcome_ori = F.softmax(self.L5(x).view(self.UEn, self.APn), dim = 1)
        UAoutcome_de = F.softmax(UAoutcome_ori * 10000, dim=1)
        UAoutcome_st = F.softmax(UAoutcome_ori, dim=1)
        PCoutcome = F.sigmoid(self.L6(x))

        return UAoutcome_ori, UAoutcome_de, UAoutcome_st, PCoutcome
