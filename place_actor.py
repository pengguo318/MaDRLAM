import torch
import torch.nn as nn
from transformer import Encoder1
from Params import configs
from agent_utils import select_action
from agent_utils import greedy_select_action
from agent_utils import sample_select_action
import numpy as np
import torch.nn.functional as F

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class place_actor(nn.Module):
    """Computing Node Selection Agent
        Output an action and the probability corresponding to the action at each scheduling step"""

    def __init__(self,
                 batch,
                 hidden_dim,
                 M,
                ):
        super().__init__()

        self.M = M

        self.hidden_dim = hidden_dim

        self.p_encoder = Encoder1(Inputdim=configs.input_dim2,
                                  embedding_size=hidden_dim,
                                  M=M).to(DEVICE)

        self.batch = batch

        self.wq = nn.Linear(hidden_dim, hidden_dim)

        self.wk = nn.Linear(hidden_dim, hidden_dim)

        self.wv = nn.Linear(hidden_dim, hidden_dim)

        self.w = nn.Linear(hidden_dim, hidden_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim)

        self.k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, index, task_op, p_action_probability, place_time, process_time, train):

        p_feas = np.concatenate((place_time.reshape(self.batch, 2, 1),
                                 process_time.reshape(self.batch, 2, 1))
                                , axis=2)

        C = 10

        p_tag = torch.LongTensor(self.batch).to(DEVICE)

        for i in range(self.batch):
            p_tag[i] = configs.n_j * i
        p_tag1 = torch.LongTensor(self.batch).to(DEVICE)
        for i in range(self.batch):
            p_tag1[i] = 2 * i

        nodes, grapha = self.p_encoder(p_feas)

        # I think rest of the method is decoder.
        # Decoder calculates the probability distribution in action space.

        torch.cuda.empty_cache()

        q = grapha

        dk = self.hidden_dim / self.M

        query = self.wq(q)  # (batch, embedding_size)

        query = torch.unsqueeze(query, dim=1)

        query = query.expand(self.batch, 2, self.hidden_dim)

        key = self.wk(nodes)

        temp = query * key

        temp = torch.sum(temp, dim=2)

        temp = temp / (dk ** 0.5)

        # Probability distribution between nodes ---------------

        temp = torch.tanh(temp) * C

        p = F.softmax(temp, dim=1)  #

        # Probability distribution between nodes ---------------

        ppp = p.view(1, -1).squeeze()

        p = torch.unsqueeze(p, dim=2)

        if train == 1:
            action_index = select_action(p)
        elif train == 2:
            action_index = sample_select_action(p)
        else:
            action_index = greedy_select_action(p)

        p_action_probability[p_tag + index] = ppp[p_tag1 + action_index]

        return action_index, p_action_probability

