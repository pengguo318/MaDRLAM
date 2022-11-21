import random
import torch
import torch.nn.functional as F
import torch.nn as nn

from Params import configs
import time

"""The encoder based on Transformer's multi-head attention layer"""

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda')
class MHA(nn.Module):
    """multi-head attention layer"""
    def __init__(self,embedding_size,M):
        super().__init__()

        self.embedding_size = embedding_size

        self.M = M

        self.dk = embedding_size / M

        self.wq = nn.Linear(embedding_size, embedding_size)

        self.wk = nn.Linear(embedding_size, embedding_size)

        self.wv = nn.Linear(embedding_size, embedding_size)

        self.w = nn.Linear(embedding_size, embedding_size)

        self.FFw1 = nn.Linear(embedding_size, embedding_size * 4)

        self.FFb1 = nn.Linear(embedding_size * 4, embedding_size)

        self.BN11 = nn.BatchNorm1d(embedding_size)

        self.BN12 = nn.BatchNorm1d(embedding_size)

    def forward(self, embedding_node):
        x_p = embedding_node

        batch = embedding_node.size()[0]

        city_size = embedding_node.size()[1]

        embedding_size = self.embedding_size

        q = self.wq(embedding_node)  # (batch,seq_len,embedding)

        q = torch.unsqueeze(q, dim=2)

        q = q.expand(batch, city_size, city_size, embedding_size)

        k = self.wk(embedding_node)  # (batch,seq_len,embedding)

        k = torch.unsqueeze(k, dim=1)

        k = k.expand(batch, city_size, city_size, embedding_size)

        v = self.wv(embedding_node)  # (batch,seq_len,embedding)

        v = torch.unsqueeze(v, dim=1)

        v = v.expand(batch, city_size, city_size, embedding_size)

        x = q * k

        x = x.view(batch, city_size, city_size, self.M, -1)

        x = torch.sum(x, dim=4)  # u=q^T x k

        x = x / (self.dk ** 0.5)

        x = F.softmax(x, dim=2)

        x = torch.unsqueeze(x, dim=4)

        x = x.expand(batch, city_size, city_size, self.M, configs.hidden_dim//self.M)

        x = x.contiguous()  #

        x = x.view(batch, city_size, city_size, -1)

        x = x * v

        x = torch.sum(x, dim=2)  # MHA :(batch, city_size, embedding_size)

        x = self.w(x)

        x = x + x_p

        #####################
        ###BN
        x = x.permute(0, 2, 1)

        x = self.BN11(x)

        x = x.permute(0, 2, 1)
        #####################
        x1 = self.FFw1(x)

        x1 = F.relu(x1,inplace=True)

        x1 = self.FFb1(x1)

        x = x + x1
        #####################
        # print('111',x.shape)
        x = x.permute(0, 2, 1)
        # print('222',x.shape)
        x = self.BN12(x)
        # print('333', x.shape)
        x = x.permute(0, 2, 1)

        return x

class Encoder1(nn.Module):
    def __init__(self,Inputdim,embedding_size,M):
        super().__init__()

        self.embedding = nn.Linear(Inputdim, embedding_size)

        self.embedding_node = embedding_size

        self.MHA = MHA(embedding_size,M)

    def forward(self,node):
        # print(node.shape)
        node = torch.from_numpy(node).to(DEVICE)

        node = self.embedding(node)

        for i in range(3):
            # MM = MHA(128,8)
            x = self.MHA(node)
            node = x

        x = node

        x = x.contiguous()

        avg = torch.mean(x, dim=1)

        return x,avg

class Encoder(nn.Module):
    def __init__(self,Inputdim,embedding_size,M):
        super().__init__()

        self.embedding = nn.Linear(Inputdim, embedding_size)

        self.embedding_node = embedding_size

        self.MHA = MHA(embedding_size,M)

    def forward(self,node):

        node = self.embedding(node)
        # print(node.shape)
        for i in range(3):

            x = self.MHA(node)

            node = x

        x = node

        x = x.contiguous()

        avg = torch.mean(x, dim=1)

        return x,avg


