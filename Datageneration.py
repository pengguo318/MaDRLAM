import numpy as np

"""Generate relevant data according to the parameters in the paper"""

def Datageneration(time,batch,n):
    """Randomly generate the datasize, deadline, cloud and edge distance of the computing task."""

    data_size = np.random.randint(1000, 2000, (time*batch,1, n))

    data_size = data_size.squeeze()

    data_size.squeeze()

    D = np.random.randint(1,10,(time*batch,1, n))

    D = D.squeeze()

    T_ = np.random.uniform(0.5, 0.75, (time*batch,1, n))

    T_ = T_.squeeze()

    return data_size,T_,D


def vvs(D,B,p,w,sita):
    """Calculate transfer speed"""

    g = np.power(D,-sita)

    a = p * g

    vs = B*1000000 * np.log2(1+a/w)

    return vs


def getdata(n_j,fill,fie,ci,B,p,w,sita,time,batch):

    fie = fie * 1000000000

    fill = np.random.randint(1, fill + 1, (n_j))

    fil = fill.astype('float32') * 100000000

    til = []
    tie = []
    tis = []
    vs = []

    taskdata = Datageneration(time,batch,n_j)

    datasize = taskdata[0]

    D = taskdata[2]

    vs = vvs(D,B,p,w,sita)

    T = taskdata[1]
    for i in datasize:

        til.append(i*ci*1000/fil)

    for i in datasize:
        tie.append(i*ci*1000/fie)

    tis = datasize*1000/vs

    return datasize,T,til,tie,tis
# getdata(n_j=10,fil=1,fie=10,ci=500,B=2,p=100,w=0.000000001,sita=4.0,time=3,batch=2)
def data(fil,fie,ci,B,p,w,sita,time,batch,n_j):

    datasizes = []

    Ts = []

    tils = []

    ties = []

    tiss = []

    getdatas = getdata(n_j,fil,fie,ci,B,p,w,sita,time,batch)

    datasizes.append(getdatas[0])

    Ts.append(getdatas[1])

    tils.append(getdatas[2])

    ties.append(getdatas[3])

    tiss.append(getdatas[-1])

    return datasizes, Ts, tils, ties, tiss

# Datageneration(2,3,20)
# a = data(fil=1,fie=10,ci=500,B=2,p=100,w=0.000000001,sita=4.0,time=3,batch=2,n_j=20,seed = 11)
# a = np.array(a)
# print(a[4].shape)
# ds = a[0].reshape((3,2,10))
# # print(ds)
# T = a[1].reshape((3,2,10))
# # print(T)
# data = np.concatenate((ds,T),axis=1)
# data = data.reshape(3,2,2,10)
# print('data',data,data.shape)
# a = np.array(a)
# a = a.reshape((3,-1,2,10))
# print(a[2])

# Datageneration(time=3,batch=2,n=10)



