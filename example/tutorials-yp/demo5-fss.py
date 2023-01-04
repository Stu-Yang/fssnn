import torch
import syft as sy
from syft.frameworks.torch.mpc.fss import le
import numpy as np

# 创建工作机
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")


# 数据
x1 = 0
x2 = torch.randint(0, 10, (1, 1))
print("x1 = ", x1, "\nx2 = ", x2)

x2_ = x2.share(bob, alice, crypto_provider=crypto_provider)
print("x2_ = ", x2_)

comp_result = le(x1, x2_)       # result of "0 <= x2_"
print(comp_result)