import torch
import torch.nn.functional as F
import syft as sy
import time
import argparse

# relu函数
def relu(self):
    return self * (self >= 0)


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


# 创建秘密份额
X = torch.rand(args.num_row, args.num_col)
protocol = "fss"
X_ = X.fix_precision(precision_fractional = 4).share(bob, alice, crypto_provider=crypto_provider, protocol=protocol)

sy.comm_total = 0           # 记录通信量
start_time = time.time()    # 记录时间

Y_ = F.relu(X_)         # 调用ReLU函数
Y = Y_.get()

print('ReLU in {}: dim of matrix = {} X {}\nTotal time is {:.3f}ms, total Communication is {:.3f}MB.\n'.format(
    protocol, args.num_row, args.num_col, (time.time()-start_time)*1000, sy.comm_total / (2**20)))
