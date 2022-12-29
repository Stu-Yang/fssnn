import torch
import torch.nn.functional as F
import syft as sy
import time

# 创建工作机
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

for n in [16, 32, 64, 128, 256, 512, 1024]:
    X = torch.rand(n, n)
    protocol = "snn"
    X_ = X.fix_precision(precision_fractional = 4).share(bob, alice, crypto_provider=crypto_provider, protocol=protocol)

    sy.comm_total = 0
    start_time = time.time()

    Y_ = F.relu(X_)
    Y = Y_.get()

    print('ReLU in {}: N = {}\nTotal time is {:.3f}ms, total Communication is {:.3f}MB.\n'.format(protocol, n, (time.time()-start_time)*1000, sy.comm_total / (2**20)))
