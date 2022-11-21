import argparse

"""This file defines some parameters of the model."""

parser = argparse.ArgumentParser(description='Arguments for task offloading')

parser.add_argument('--device', type=str, default="cuda", help='Device')

parser.add_argument('--n_j', type=int, default=10, help='Number of jobs of the instance')

parser.add_argument('--maxtask', type=int, default=2, help='Maximum number of parallel computations in the cloud')

parser.add_argument('--batch', type=int, default=24, help='batch size')

parser.add_argument('--time', type=int, default=2500, help='training times per epoch')

parser.add_argument('--testtime', type=int, default=2, help='test')

parser.add_argument('--comtesttime', type=int, default=1, help='vali')

parser.add_argument('--fil', type=int, default=5, help='The CPU computing capacity of the edge device')

parser.add_argument('--fie', type=int, default=15, help='The CPU computing capacity of the cloud')

parser.add_argument('--ci', type=int, default=500, help='The CPU cycles to compute one bit of data')

parser.add_argument('--B', type=int, default=2, help='The communication bandwidth between cloud and edge device')

parser.add_argument('--p', type=int, default=100, help='The transmit power of the edge device')

parser.add_argument('--w', type=int, default=0.000000001, help='the variance of complex white Gaussian channel noise')

parser.add_argument('--Men', type=int, default=100000, help='Menmory')

parser.add_argument('--sita', type=int, default=4.0, help='the path-loss exponent')

parser.add_argument('--input_dim1', type=int, default=3, help='Inout dim of task selection agent')

parser.add_argument('--input_dim2', type=int, default=2, help='Inout dim of computing node selection agent')

parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dim')

parser.add_argument('--epochs', type=int, default=30, help='No. of episodes of each env for training')

configs = parser.parse_args()
