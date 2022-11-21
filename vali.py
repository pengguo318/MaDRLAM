from act_critic import actor_critic
import sys
import importlib
from Params import configs
import torch
import numpy as np
import datetime

"""Load the trained model for testing"""

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print(DEVICE)

starttime =  datetime.datetime.now()

compare = 1

maxtask = configs.maxtask

size = '1000_2000'

net1 = actor_critic(batch=configs.batch,
                    hidden_dim = configs.hidden_dim,
                    M=8,
                    device=configs.device).to(DEVICE)

testdatas = np.load('data2//{}//compare{}//com_testdatas{}_{}.npy'.format(configs.n_j,compare,configs.n_j,size))

print(testdatas.shape)

length = torch.zeros(1).to(DEVICE)

path = './train_process/{}/compare{}/actor{}_mutil_actor.pt'.format(configs.n_j,compare,configs.n_j)

net1.load_state_dict(torch.load(path, DEVICE))

importlib.reload(sys)

with torch.no_grad():

    for j in range(configs.comtesttime):

        torch.cuda.empty_cache()

        seq, seq2, task_action_pro, _, r = net1(testdatas[j], 0)

        length = length + torch.mean(r)

        print('task order', seq[0])

        print('computing node', seq2[0])

        print(r)

    length = length / configs.comtesttime  ##avg

    print('time_delay =', length)

    endtime = datetime.datetime.now()

    print('runingtime=', (endtime - starttime) / configs.batch)
