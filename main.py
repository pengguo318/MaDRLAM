from Params import configs

import torch
import os
import numpy as np
from act_critic import actor_critic
from scipy.stats import ttest_rel

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

compare = 1

size = '1000_2000'

"""Load training data and test data"""

datas = np.load('.//data2//{}//compare{}//datas{}_{}.npy'.format(configs.n_j, compare, configs.n_j, size))

datas.astype('float16')

print(datas.dtype)

testdatas = np.load('.//data2//{}//compare{}//com_testdatas{}_{}.npy'.format(configs.n_j,compare,configs.n_j,size))

Net1 = actor_critic(batch=configs.batch,
                    hidden_dim=configs.hidden_dim,
                    M=8,
                    device=configs.device).to(DEVICE)

Net2 = actor_critic(batch=configs.batch,
                    hidden_dim=configs.hidden_dim,
                    M=8,
                    device=configs.device).to(DEVICE)

Net2.place_actor.load_state_dict(Net1.place_actor.state_dict())

min = 50000000000

if configs.batch == 24:
    lr = 0.000005
    print('lr=', lr)

elif configs.batch == 8:
    lr = 0.0000005
    print('lr=', lr)

bl_alpha = 0.05

output_dir = 'train_process//{}//compare{}'.format(configs.n_j, compare)

save_dir = os.path.join(os.getcwd(), output_dir)

contintrain = 0

Net2.load_state_dict(Net1.state_dict())

for epoch in range(configs.epochs):
    for i in range(configs.time):
        data = datas[i]

        # Getting information about env
        task_seq, p_seq, task_action_pro, p_action_pro, reward1, load_balancing_eff, energy_consumption = Net1(data, 1)

        _, _, _, _, reward2, _, _ = Net2(data, 1)

        reward1 = reward1.detach()

        torch.cuda.empty_cache()

        # Update networks information found in line 73
        Net1.updata(task_action_pro, reward1, reward2, lr)
        Net1.updata2(p_action_pro, reward1, reward2, lr)

        print('epoch={},i={},time1={},time2={}'.format(epoch, i, torch.mean(reward1),
                                                       torch.mean(reward2)))

        with torch.no_grad():

            if (reward1.mean() - reward2.mean()) < 0:

                tt, pp = ttest_rel(reward1.cpu().numpy(), reward2.cpu().numpy())

                p_val = pp / 2

                assert tt < 0, "T-statistic should be negative"

                if p_val < bl_alpha:
                    print('Update baseline')

                    Net2.load_state_dict(Net1.state_dict())

            """Every 20 iterations check whether the model needs to save parameters"""

            if i % 20 == 0:

                length = torch.zeros(1).to(DEVICE)

                for j in range(configs.comtesttime):
                    torch.cuda.empty_cache()

                    _, _, _, _, r, _, _ = Net1(testdatas[j], 0)

                    length = length + torch.mean(r)

                length = length / configs.comtesttime  ##

                if length < min:
                    torch.save(Net1.state_dict(), os.path.join(save_dir,
                                                               'epoch{}-i{}-dis_{:.5f}.pt'.format(
                                                                   epoch, i, length.item())))

                    torch.save(Net1.state_dict(), os.path.join(save_dir,
                                                               'actor{}_mutil_actor.pt'.format(configs.n_j)))

                    min = length
                print(os.getcwd())
                # file_writing_obj1 = open('/content/MaDRLAM/lr_change/lr_000005/train_vali/{}/compare{}/{}_{}.txt'.format(configs.n_j, compare,configs.n_j,configs.maxtask),
                #                          'a')

                file_writing_obj1 = open('./lr_000005/{}_{}.txt'.format(configs.n_j, configs.maxtask),
                                         'a')

                file_writing_obj1.writelines(str(length) + '\n')

                print('length=', length.item(), 'min=', min.item())

                file_writing_obj1.close()

                load_balancing_eff_writing_obj = open('./lb/{}_{}.txt'.format(configs.n_j, configs.maxtask),
                                         'a')

                load_balancing_eff_writing_obj.writelines(str(load_balancing_eff) + '\n')

                energy_consumption_writing_obj = open('./ec/{}-{}.txt'.format(configs.n_j, configs.maxtask),
                                         'a')
                energy_consumption_writing_obj.writelines(str(energy_consumption) + '\n')
