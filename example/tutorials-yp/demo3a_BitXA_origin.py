import torch
import syft as sy

# 创建工作机
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

n = 32      # bit length
N = 784     # matrix dim

X = torch.randint(0, 2**n, (N, N))
Y = torch.randint(0, 2, (N, 1))       # 直接计算则必须是同类型
# Y = torch.randint(0, 2, (N, 1)).type(torch.ByteTensor)

X_ = X.share(bob, alice, crypto_provider=crypto_provider)
Y_ = Y.share(bob, alice, crypto_provider=crypto_provider)


sy.comm_total = 0           # commuication

C_ = X_.matmul(Y_)
C = C_.get()

comm_total = sy.comm_total

print("bit length = {} bits, and dim of matrix is {} * {}\n Total communication is {:.5f} MB".format(n, N, N, comm_total / 2**20))


# bit length = 32 bits, and dim of matrix is 784 * 784
# Total communication is 9.40368 MB
# 根据结果显示，PySyft可以接受不同数据类型的tensor进行矩阵相乘，其原因在于.share(bob, alice, crypto_provider=crypto_provider)中会将其都转换为AdditiveTensor