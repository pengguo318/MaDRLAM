from torch.distributions.categorical import Categorical


"""Random selection strategy and greedy selection strategy"""

def select_action(p):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return s

def greedy_select_action(p):##(batch,n_j*2,1)
    _, index = p.squeeze().max(1)
    # action = candidate[index]
    return index

def sample_select_action(p):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return s

