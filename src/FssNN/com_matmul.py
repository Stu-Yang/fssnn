
import torch
import syft as sy
import time
import math
import numpy as np
if __name__ == "__main__":
    # 创建工作机
    hook = sy.TorchHook(torch)
    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
    sy.local_worker.crypto_store.layers = 3    # 表示linear的层数。大于等于9为不相关;测试优化则设置为3
    batch_sizes=[64,128]    # 可设置多种 batch_size
    col_row_s=[784,128,128];row_s=[128,128,10]
    for batch_size in batch_sizes:
        sy.comm_total = 0   # commuication
        sy.comm_matmul = 0  # me向alice、bob分发beaver时产生的通信量
        sy.local_worker.crypto_store.clear_matmul_turple()
        comm = []
        X_s=[];W_s=[]
        δ_s=[]
        matrix_dim = []
        matmul_a_b=[]
        loc=1
        layers=3
        for i in np.arange(3):
            X_s.append(torch.randint(0, 2 ** 16 - 1, (batch_size, col_row_s[i])))
            W_s.append(torch.randint(0, 2 ** 16-1, (col_row_s[i],row_s[i])))
            δ_s.append(torch.randint(0, 2 ** 16-1, (batch_size, row_s[2-i])))
        while loc<3*layers:
            if loc == 1:
                matmul_a_b.clear()
                matmul_a_b.append([])
            a_b_loc = -1
            if loc > layers:  # printf
                loc_δ_s = math.ceil((loc - layers) / 2)-1  # 0,1,...,layers-1
                a_b_loc = (loc % 2) ^ (layers % 2)  # 0表示用a，1表示重用b  #相关矩阵
                if a_b_loc==1:
                    A=δ_s[loc_δ_s];B=W_s[layers-loc_δ_s-1].t()
                else:
                    A=X_s[layers-loc_δ_s-1].t();B=δ_s[loc_δ_s]
            else:
                A =X_s[loc - 1];B=W_s[loc - 1]
            shape=[A.shape,B.shape]
            A_ = A.share(bob, alice, crypto_provider=crypto_provider, protocol="fss")
            B_ = B.share(bob, alice, crypto_provider=crypto_provider, protocol="fss")

            # seucre multiplication
            time_start = time.time()  # time
            E_ = A_.matmul(B_)
            E = E_.get()

            comm_total = sy.comm_total
            comm.append(comm_total)
            loc+=1

        print("Batch size={}: Total communication is {:.5f} MB, matmul communication is {:.5f} MB, and total time is {:.5f} s.".format(
                batch_size,
                ((comm_total) / 2 ** 20),
                (sy.comm_matmul) / 2**20, 
                (time.time() - time_start)))
        del sy.comm_total
        del sy.comm_matmul
