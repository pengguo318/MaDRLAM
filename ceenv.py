import gym
import numpy as np
import torch
from gym.utils import EzPickle
from Params import configs

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

"""Environment for agent interaction, covering feature extraction and update"""

BREAKING_FACTOR = 0.1

INITIAL_ENERGY = 20_000
PROCESS_ENERGY_FACTOR = 2
OFFLOAD_ENERGY_FACTOR = 1


class CLOUD_edge(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 maxtasks):
        EzPickle.__init__(self)
        self.maxtasks = maxtasks

        self.n_j = n_j

        self.step_count = 0

        self.L = 50  # penalty value

        self.number_of_jobs = n_j

        self.number_of_tasks_on_cloudy = 0

        self.busy_men_on_cloudy = 0

    def reset(self, batch, data):
        """initialization"""
        self.batch = batch
        self.job_finish_time_on_cloudy = np.zeros(self.batch * self.maxtasks).reshape((self.batch, -1))

        self.step_count = 0
        # print(self.step_count)

        # self.place = data[-1]  ##

        self.datasize = np.array(data[0], dtype=np.single)
        self.deadline = np.array(data[1], dtype=np.single)
        self.dur_l = np.array(data[2], dtype=np.single)  # single  ##
        self.dur_e = np.array(data[3], dtype=np.single)
        self.dur_s = np.array(data[4], dtype=np.single)

        # Extracting task feature
        self.I = np.full(shape=(self.batch, self.n_j, 2), fill_value=0, dtype=bool)

        self.LBs = np.zeros((self.batch, self.n_j, 2), dtype=np.single)

        self.Fi = np.zeros((self.batch, self.n_j, 2), dtype=np.single)

        self.LBm = np.zeros((self.batch, self.n_j, 1), dtype=np.single)

        self.Fim = np.zeros((self.batch, self.n_j, 1), dtype=np.single)

        # self.G_LBs = np.ones((self.batch,self.n_j,2), dtype=np.single)
        self.place_time = np.zeros((self.batch, 2), dtype=np.single)

        self.task_mask = np.full(shape=self.deadline.shape, fill_value=0, dtype=bool)

        self.place_mask = np.full(shape=self.LBs.shape, fill_value=0, dtype=bool)

        # TODO: this is not correct, cloud node's initial energy is not equal to local node's energy
        self.edges_energies = np.full((self.n_j, 2), fill_value=INITIAL_ENERGY)

        # REms stands for Reduced Energy minimum
        self.REm = np.zeros((self.batch, self.n_j), dtype=np.single)

        # self.Fi = np.zeros((self.batch,self.n_j,2), dtype=np.single)
        for i in range(self.batch):
            for j in range(self.n_j):
                # on edge
                self.LBs[i][j][0] = self.dur_l[i][j]

                # on cloud
                self.LBs[i][j][1] = self.dur_s[i][j] + self.dur_e[i][j]

                self.Fi[i][j][0] = self.deadline[i][j] - self.LBs[i][j][0]

                self.Fi[i][j][1] = self.deadline[i][j] - self.LBs[i][j][1]

                self.LBm[i][j][0] = min(self.LBs[i][j][0], self.LBs[i][j][1])

                self.Fim[i][j][0] = self.Fi[i][j][1]


        task_features = np.concatenate((
                                        self.LBm.reshape(self.batch, self.n_j, 1),
                                        self.Fim.reshape(self.batch, self.n_j, 1),
                                        self.task_mask.reshape(self.batch, self.n_j, 1),
                                        self.REm.reshape(self.batch, self.n_j, 1),
                                        )
                                       , axis=2)

        # print(self.I[0])
        return task_features, self.task_mask, self.place_time

    def step(self, task_action, p_action, tasks_per_node):
        """Update features based on the actions of the agents"""
        total_energy_consumption = 0
        energies = [PROCESS_ENERGY_FACTOR, OFFLOAD_ENERGY_FACTOR]

        for i in range(self.batch):
            if p_action[i] == 1:
                earliest_ind = np.argmin(self.job_finish_time_on_cloudy[i])

                self.job_finish_time_on_cloudy[i][earliest_ind] = self.LBs[i][task_action[i]][1]

                min_ind = np.argmin(self.job_finish_time_on_cloudy[i])

                self.place_time[i][1] = self.job_finish_time_on_cloudy[i][min_ind]

            tasks_per_node[p_action[i] * 10 + task_action[i]] += 1

        # compute reward
        reward = np.zeros((self.batch, 1))
        for i in range(self.batch):
            selected_node = task_action[i]
            correct_energy_decision = False

            energy_consumption = self.datasize[i][selected_node] * energies[p_action[i]]

            #  if the task meets deadline or not
            if self.LBs[i][selected_node][p_action[i]] <= self.deadline[i][selected_node]:
                # reducing process energy from selected edge
                if self.edges_energies[selected_node][p_action[i]] >= energy_consumption:
                    correct_energy_decision = True

            if correct_energy_decision:
                self.edges_energies[selected_node][p_action[i]] -= energy_consumption
                total_energy_consumption = total_energy_consumption + energy_consumption
                reward[i] = self.LBs[i][selected_node][p_action[i]]
            else:
                reward[i] = self.LBs[i][selected_node][p_action[i]] * 10
                # print('timewindows')

        print('reward', reward[0])
        earliest_time = np.zeros((self.batch, 1))
        for i in range(self.batch):
            earliest_time[i] = min(self.job_finish_time_on_cloudy[i])

        for i in range(self.batch):
            self.I[i][task_action[i]][0] = True

            self.I[i][task_action[i]][1] = True

        for b in range(self.batch):
            self.task_mask[b][task_action[b]] = True

        for i in range(self.batch):
            for j in range(self.n_j):

                deadline = self.T[i][j] * BREAKING_FACTOR

                if self.I[i][j][0] == False and self.I[i][j][1] == False:
                    # EDGE
                    job_ready_time_a_e = 0
                    compute_ready_time_a_e = 0

                    job_start_time_a_e = max(job_ready_time_a_e, compute_ready_time_a_e)

                    # dur_l is the processing time of task i in core j in edge layer
                    job_finish_time_a_e = job_start_time_a_e + self.dur_l[i][j]
                    # dur_l is the processing time of task i in core j
                    job_busy_time_a_e = np.sum(self.LBs[:, j, 0])

                    job_finish_time_a_e = job_busy_time_a_e + job_start_time_a_e + self.dur_l[i][j]

                    self.LBs[i][j][0] = job_finish_time_a_e * BREAKING_FACTOR

                    if deadline < self.LBs[i][j][0]:
                        self.Fi[i][j][0] = 0
                        # self.LBs[i][j][0] = self.T[i][j]
                    else:
                        self.Fi[i][j][0] = self.T[i][j] - self.LBs[i][j][0]

                    # CLOUD
                    job_ready_time_a_c = self.dur_s[i][j]
                    compute_ready_time_a_c = min(self.job_finish_time_on_cloudy[i])

                    job_start_time_a_c = max(job_ready_time_a_c, compute_ready_time_a_c)

                    job_busy_time_a_c = np.sum(self.LBs[:, j, 1])

                    job_finished_time_a_c = job_busy_time_a_c + job_start_time_a_c + self.dur_e[i][j]

                    # calculating remained time to deadline
                    self.Fi[i][j] = self.deadline[i][j] - self.LBs[i][j]
                    self.LBs[i][j][1] = job_finished_time_a_c * BREAKING_FACTOR

                    if deadline < self.LBs[i][j][1]:
                        # If finish time exceeds deadline, put finish time equal to deadline
                        # self.LBs[i][j][1] = self.T[i][j]
                        self.Fi[i][j][1] = 0
                    else:
                        self.Fi[i][j][1] = self.T[i][j] - self.LBs[i][j][1]

                    # choosing between edge and cloud
                    # based on the time that the task is finished
                    self.LBm[i][j][0] = min(self.LBs[i][j][0], self.LBs[i][j][1])

                    # is this ok ?? It chooses the Fi on cloud
                    self.Fim[i][j][0] = self.Fi[i][j][1]

                    # Energy features
                    self.REm[i][j] = min(INITIAL_ENERGY - self.edges_energies[j][0], INITIAL_ENERGY - self.edges_energies[j][1])

        task_feas = np.concatenate((self.LBm.reshape(self.batch, self.n_j, 1),
                                    self.Fim.reshape(self.batch, self.n_j, 1),
                                    self.task_mask.reshape(self.batch, self.n_j, 1),
                                    self.REm.reshape(self.batch, self.n_j, 1)
                                    )
                                   , axis=2)

        # print('LBs',self.LBs[0])
        # print('F',self.Fi[0])

        # print(self.task_mask[0])
        return task_feas, self.task_mask, self.place_time, reward, tasks_per_node, total_energy_consumption


"""test"""
# env = CLOUD_edge(n_j=configs.n_j,
#                               maxtasks=configs.maxtask,
#                               max_Men=configs.Men)
# datas = np.load('data2//{}//compare{}//datas{}_{}.npy'.format(configs.n_j,1,configs.n_j,'1000_2000'))
# data = datas[0]
# task_feas,task_mask,place_time = env.reset(10, data)
# task = np.zeros((10,24),dtype=np.single)
# # task = task.float(32)
# task[0] = 0
# task[1],task[2],task[3],task[4],task[5],task[6],task[7],task[8],task[9] = 1,2,3,4,5,6,7,8,9
# task =task.astype(int)
# place = np.ones((10,24),dtype=np.single).astype(int)
# # print(task[0].reshape(24))
#
# dur_l = np.array(data[2], dtype=np.single)  # single  ##
# dur_e = np.array(data[3], dtype=np.single)
# dur_s = np.array(data[4], dtype=np.single)
# T = np.array(data[1], dtype=np.single)
# print('edge:',dur_l[0])
# print('trans',dur_s[0])
# print('cloud',dur_e[0])
# print('deadline',T[0])
# for i in range(10):
#
#     task_feas, task_mask, place_time, reward = env.step(task[i].reshape(24), place[i].reshape(24))
