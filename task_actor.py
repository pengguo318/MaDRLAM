import torch.nn as nn
from Params import configs
from transformer import Encoder1
import torch
from agent_utils import select_action
from agent_utils import greedy_select_action
from agent_utils import sample_select_action
import numpy as np
import torch.nn.functional as F

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class task_actor(nn.Module):
    """Task Selection Agent;
    Output an action and the probability corresponding to the action at each scheduling step."""
    def __init__(self,
                 batch,
                 hidden_dim,
                 M):
        super().__init__()
        self.M = M

        self.hidden_dim = hidden_dim

        self.task_encoder = Encoder1(Inputdim=configs.input_dim1,
                                     embedding_size=configs.hidden_dim,
                                     M=M).to(DEVICE)

        self.batch = batch

        self.wq = nn.Linear(hidden_dim, hidden_dim)

        self.wk = nn.Linear(hidden_dim, hidden_dim)

        self.wv = nn.Linear(hidden_dim, hidden_dim)

        self.w = nn.Linear(hidden_dim, hidden_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim)

        self.k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data, job_index, feas, mask, action_pro, train):
        """
        This method determines that for the specific job, the probability
        of which task is higher.
        """

        mask = torch.from_numpy(mask).to(DEVICE)

        tag = torch.LongTensor(self.batch).to(DEVICE)
        for i in range(self.batch):
            tag[i] = configs.n_j * i

        C = 10

        nodes, grapha = self.task_encoder(feas)

        torch.cuda.empty_cache()

        q = grapha

        dk = self.hidden_dim / self.M

        query = self.wq(q)  # (batch, embedding_size)

        query = torch.unsqueeze(query, dim=1)

        query = query.expand(self.batch, configs.n_j, self.hidden_dim)

        key = self.wk(nodes)

        temp = query * key

        temp = torch.sum(temp, dim=2)

        temp = temp / (dk ** 0.5)

        # Probability distribution between nodes ---------------
        temp = torch.tanh(temp) * C

        temp.masked_fill_(mask, float('-inf'))

        # "p" stands for probability
        # the shape of p is (24,10)
        p = F.softmax(temp, dim=1)

        # -----------------------------------------------------

        ppp = p.view(1, -1).squeeze()

        p = torch.unsqueeze(p, dim=2)

        if train == 1:
            action_index = select_action(p)
        elif train == 2:
            action_index = sample_select_action(p)
        else:
            action_index = greedy_select_action(p)

        action_pro[tag + job_index] = ppp[tag + action_index]
        # action_index.shape = (24,1) - number of tasks - for each task it finds suitable job

        datasize = np.array(data[0], dtype=np.single)
        T = np.array(data[1], dtype=np.single)
        dur_l = np.array(data[2], dtype=np.single)  # single  ##
        dur_e = np.array(data[3], dtype=np.single)
        dur_s = np.array(data[4], dtype=np.single)

        # process_time is a PyTorch array that stores process time for each task based on
        # it's corresponding job (machine).
        process_time = np.zeros((self.batch, 2), dtype=np.single)

        for i in range(self.batch):
            process_time[i][0] = dur_l[i][action_index[i]]

            process_time[i][1] = dur_e[i][action_index[i]]

        return action_index, action_pro, process_time  # (batch,1)
