import torch
from torch_geometric.nn import ChebConv
import torch.nn.functional as F
import torch.nn as nn


class EMOGINet(torch.nn.Module):
    def __init__(self,args):
        super(EMOGINet, self).__init__()
        self.args = args
        self.conv1 = ChebConv(50, 128, K=2)
        self.conv2 = ChebConv(128, 128, K=2)
        self.conv3 = ChebConv(128, 1, K=2)

    def forward(self, data):
        edge_index = data.edge_index
        x = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x
