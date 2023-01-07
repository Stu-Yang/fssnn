import torch
import numpy as np
import time
import argparse
import syft as sy

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--num_row", help="the number of rows of the matrix", type=int)
parser.add_argument("--num_col", help="the number of columns of the matrix", type=int)
args = parser.parse_args()

# 创建工作机
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")


# 数据
x1 = torch.zeros(args.num_row, args.num_col)
x2 = torch.rand(args.num_row, args.num_col)
# print("x1 = ", x1, "\nx2 = ", x2)

protocol = "fss"
x1_ = x1.fix_precision(precision_fractional = 4).share(bob, alice, crypto_provider=crypto_provider, protocol=protocol)
x2_ = x2.fix_precision(precision_fractional = 4).share(bob, alice, crypto_provider=crypto_provider, protocol=protocol)

sy.comm_total = 0           # 记录通信量
start_time = time.time()    # 记录时间

comp_result = x2_ * (x1_ <= x2_)       # result of "x2_ * (0 <= x2_)"
# print(comp_result)

print('ReLU in {}: dim of matrix = {} X {}\nTotal time is {:.3f}ms, total Communication is {:.3f}MB.\n'.format(
    protocol, args.num_row, args.num_col, (time.time()-start_time)*1000, sy.comm_total / (2**20)))
