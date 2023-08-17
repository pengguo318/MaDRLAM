import numpy as np
from matplotlib import pyplot as plt

img_array = np.load("data2/10/compare1/datas10_1000_2000.npy")

print(img_array.shape)
print(img_array[0, :, 6, :])