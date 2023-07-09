from Params import configs
from ceenv import CLOUD_edge
import torch.nn as nn
from transformer import Encoder1
import torch
from torch import optim
from task_actor import task_actor
from place_actor import place_actor

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

"""This part covers the two agents proposed in the article (Task Selection Agent and Computing Node Selection Agent"""

class actor_critic(nn.Module):
    """Two agents work together to obtain scheduling results"""

    def __init__(self,
                 batch,
                 hidden_dim,
                 M,
                 device):

        super().__init__()

        self.M = M

        self.hidden_dim = hidden_dim

        self.env = CLOUD_edge(n_j=configs.n_j,
                              maxtasks=configs.maxtask,
                              max_Men=configs.Men)

        self.task_actor = task_actor(batch=batch,
                                     hidden_dim=hidden_dim,
                                     M=M)

        self.place_actor = place_actor(batch=batch,
                                       hidden_dim=hidden_dim,
                                       M=M)

        self.batch = batch

    # this method is being called in the main.py
    def forward(self, data, train):

        action_pro = torch.zeros(self.batch, configs.n_j).to(DEVICE).view(1, -1).squeeze().to(DEVICE)

        place_action_pro = torch.zeros(self.batch, configs.n_j).to(DEVICE).view(1, -1).squeeze().to(DEVICE)

        task_feas, task_mask, place_time = self.env.reset(self.batch, data)

        task_seq_list = []

        p_op_list = []

        rewards = 0

        q = torch.zeros((self.batch, 1)).to(DEVICE)

        for i in range(configs.n_j):
            index = i

            task_action, action_pro, process_time = self.task_actor(data, index, task_feas, task_mask, action_pro,
                                                                train)  ##选择任务
            # I think task_op determines which task has the most priority
            ind = torch.unsqueeze(task_action, 1).tolist()

            task_seq_list.append(task_action)

            place_action, place_action_pro = self.place_actor(index, task_action, place_action_pro, place_time, process_time, train)

            p_op_list.append(place_action)

            task_feas, task_mask, place_time, reward = self.env.step(task_action, place_action)

            rewards += reward

        place_action_pro = place_action_pro.view(self.batch, configs.n_j)  ##(batch,n)

        task_action_pro = action_pro.view(self.batch, configs.n_j)  ##(batch,n)！！！！！

        task_seq = torch.unsqueeze(task_seq_list[0], 1)

        place_seq = torch.unsqueeze(p_op_list[0], 1)

        q = q.to(DEVICE)

        for i in range(configs.n_j - 1):
            task_seq = torch.cat([task_seq, torch.unsqueeze(task_seq_list[i + 1], 1)], dim=1)
        for i in range(configs.n_j - 1):
            place_seq = torch.cat([place_seq, torch.unsqueeze(p_op_list[i + 1], 1)], dim=1)
        # print(task_seq[0])

        rewards = torch.from_numpy(rewards).to(DEVICE)

        rewards = rewards.to(torch.float32)

        return task_seq, place_seq, task_action_pro, place_action_pro, rewards

    # Updates actor 1 nn
    def updata(self, task_action_pro, reward1, q, lr):# q is the reward2

        opt = optim.Adam(self.task_actor.parameters(), lr)

        pro = torch.log(task_action_pro)

        loss = torch.sum(pro, dim=1)

        score = reward1 - q

        score = score.detach()

        loss = score * loss

        loss = torch.sum(loss) / configs.batch

        opt.zero_grad()
        loss.backward()
        opt.step()

    # Updates actor 2 nn
    def updata2(self, p_action_pro, reward1, q, lr):

        opt = optim.Adam(self.place_actor.parameters(), lr)

        pro = torch.log(p_action_pro)

        loss = torch.sum(pro, dim=1).to(DEVICE)

        score = reward1 - q

        score = score.detach().to(DEVICE)

        loss = score * loss

        loss = torch.sum(loss) / configs.batch

        opt.zero_grad()
        loss.backward()
        opt.step()
