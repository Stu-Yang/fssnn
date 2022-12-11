# There are some basic operations about PySyft, refer to function "build_prepocessing()" in yp_workplace/ariann/ariann/preprocess.py
# Some Goals:
# 1. Addtion and multiplication in the context of secure computation (2PC)
# 2. Matrix addition and matrix multiplication in the context of secure computation (2PC)
# 3. Measure some metrics about secure computation, such as total time and commuiciation info


import torch
import syft as sy
import time
import numpy as np
import matplotlib.pyplot as plt

# 创建工作机
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")


# ------------------------------------------------------------------
# Goal 1&3: secure addition and multiplication
# Compute c = a * b

# a ,b = [5], [25]

# a_ = torch.tensor(a).share(bob,alice, crypto_provider=crypto_provider)
# b_ = torch.tensor(b).share(bob,alice, crypto_provider=crypto_provider)

# # seucre multiplication
# sy.comm_total = 0           # commuication
# time_start = time.time()    # time

# c = a_ * b_
# c.get()

# print("secure multiplication:\nTotal communication is {:.3f} MB.\nTotal time is {:.3f} s."
#         .format(((sy.comm_total) / 10 ** 6), (time.time() - time_start)))
# del sy.comm_total

# ------------------------------------------------------------------
# Goal 2&3: seucre matrix addition and multiplication
# Compute C = A X B

comm = []
matrix_dim = []

for n in range(1, 7):
    row, col = 16 * (2 ** n), 16 * (2 ** n)
    matrix_dim.append(row)

    A = torch.randint(0, 2**16-1, (row, col))
    B = torch.randint(0, 2**16-1, (row, col))

    A_ = A.share(bob,alice, crypto_provider=crypto_provider)
    B_ = B.share(bob,alice, crypto_provider=crypto_provider)

    # seucre multiplication
    sy.comm_total = 0           # commuication
    time_start = time.time()    # time

    C = A_.mm(B_)               # mm() is secure matrix multiplication, is quivalent to matmul()
                                # and is defined in ~/gsq_workplace/arinn/PySyft/syft/frameworks/torch/tensors/interpreters/additive_shared.py line 679
    C.get()

    comm_total = sy.comm_total
    comm.append(comm_total)

    print("matrix dimension = {} : Total communication is {:.5f} MB, and total time is {:.5f} s in secure multiplication."
            .format(row, ((comm_total) / 2**20), (time.time() - time_start)))           # 10 ** 6 in yp_workplace/ariann/ariann/preprocess.py line 237 instead of 2 ** 20
    del sy.comm_total

# 输出显示矩阵维度增加一倍（*2），通信量增加两倍（*4），即通信量和矩阵维度的平方成正比，这和理论结果是相符的。但是，计算时间是有些异常的，可能和计算机内部的矩阵乘法优化有关
# Output:
# matrix dimension = 32 : Total communication is 0.04591 MB, and total time is 0.85050 s in secure multiplication.
# matrix dimension = 64 : Total communication is 0.16310 MB, and total time is 0.71070 s in secure multiplication.
# matrix dimension = 128 : Total communication is 0.63186 MB, and total time is 0.80319 s in secure multiplication.
# matrix dimension = 256 : Total communication is 2.50687 MB, and total time is 0.90806 s in secure multiplication.
# matrix dimension = 512 : Total communication is 10.00687 MB, and total time is 1.02344 s in secure multiplication.
# matrix dimension = 1024 : Total communication is 40.00687 MB, and total time is 2.12527 s in secure multiplication.