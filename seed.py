import numpy as np
import numpy as np
from Params import configs
# from seed import seed

from Datageneration import data
"""This part is used to generate the training set, test set and verification set required for model training."""
def traindata():
    """Generate a training dataset for model training"""
    datas = datatestdatas = data(configs.fil,
                                 configs.fie,
                                 configs.ci,
                                 configs.B,
                                 configs.p,
                                 configs.w,
                                 configs.sita,
                                 configs.time,
                                 configs.batch,
                                 configs.n_j,
                                 )
    # print(data.shape)
    datas = np.array(datas)
    # print(datas[0].shape)
    ds = datas[0].reshape((configs.time, configs.batch, configs.n_j))
    T = datas[1].reshape((configs.time, configs.batch, configs.n_j))
    tils = datas[2].reshape((configs.time, configs.batch, configs.n_j))
    # print(tils[0])
    ties = datas[3].reshape((configs.time, configs.batch, configs.n_j))
    # print(ties.shape)
    tiss = datas[4].reshape((configs.time, configs.batch, configs.n_j))
    datas = np.concatenate((ds, T, tils, ties, tiss), axis=1)
    datas = datas.reshape(configs.time, -1, configs.batch, configs.n_j)
    datas = list(datas)
    # print(datas)

    np.save('datas{}_1000_2000.npy'.format(configs.n_j), datas)

def data2():
    """Generate a test dataset for model training"""
    testdatas = data(configs.fil,
                     configs.fie,
                     configs.ci,
                     configs.B,
                     configs.p,
                     configs.w,
                     configs.sita,
                     configs.testtime, configs.batch, configs.n_j,
                     )
    testdatas = np.array(testdatas)
    ds = testdatas[0].reshape((configs.testtime, configs.batch, configs.n_j))
    T = testdatas[1].reshape((configs.testtime, configs.batch, configs.n_j))
    tils = testdatas[2].reshape((configs.testtime, configs.batch, configs.n_j))
    ties = testdatas[3].reshape((configs.testtime, configs.batch, configs.n_j))
    tiss = testdatas[4].reshape((configs.testtime, configs.batch, configs.n_j))
    testdatas = np.concatenate((ds, T, tils, ties, tiss), axis=1)
    testdatas = testdatas.reshape(configs.testtime, -1, configs.batch, configs.n_j)

    testdatas = list(testdatas)
    # for i in range(configs.testtime):
    #     datas.append(data(configs.batch, configs.n_j))
    # np.save('testdatas13_1000_2000.npy', testdatas)
    np.save('testdatas{}_1000_2000.npy'.format(configs.n_j), testdatas)


def data3():
    """Generate a validation dataset for model training"""
    testdatas = data(configs.fil,
                     configs.fie,
                     configs.ci,
                     configs.B,
                     configs.p,
                     configs.w,
                     configs.sita,
                     configs.comtesttime, configs.batch, configs.n_j,
                     )
    testdatas = np.array(testdatas)
    ds = testdatas[0].reshape((configs.comtesttime, configs.batch, configs.n_j))
    T = testdatas[1].reshape((configs.comtesttime, configs.batch, configs.n_j))
    tils = testdatas[2].reshape((configs.comtesttime, configs.batch, configs.n_j))
    ties = testdatas[3].reshape((configs.comtesttime, configs.batch, configs.n_j))
    tiss = testdatas[4].reshape((configs.comtesttime, configs.batch, configs.n_j))
    testdatas = np.concatenate((ds, T, tils, ties, tiss), axis=1)
    testdatas = testdatas.reshape(configs.comtesttime, -1, configs.batch, configs.n_j)

    #     datas.append(data(configs.batch, configs.n_j))

    # np.save('com_testdatas13_1000_2000.npy', testdatas)
    np.save('com_testdatas{}_1000_2000.npy'.format(configs.n_j), testdatas)




np.random.seed(11)
# traindata()
# data2()
data3()