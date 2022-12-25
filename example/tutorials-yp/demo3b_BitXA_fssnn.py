import torch
import syft as sy

# 创建工作机
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

# 一些参数
n = 32      # bit length
N = 784     # matrix dim

# 矩阵逐元素异或
def MatXor(X, Y):
    return X + Y - 2 * torch.mul(X, Y)

# 算术秘密分享
def Additive_A(X, workers):
    P0, P1 = workers[0], workers[1]
    row, col = X.shape[0], X.shape[1]

    R = torch.randint(0, 2**n, (row, col))  # 随机矩阵
    X0, X1 = X-R, R
    X0_ptr = X0.send(P0)
    X1_ptr = X1.send(P1)

    return X0_ptr, X1_ptr

# 布尔秘密分享
def Additive_B(X, workers):
    P0, P1 = workers[0], workers[1]
    row, col = X.shape[0], X.shape[1]

    R = torch.randint(0, 2, (row, col)).type(torch.ByteTensor)  # 随机矩阵
    X0, X1 = MatXor(X, R), R
    X0_ptr = X0.send(P0)
    X1_ptr = X1.send(P1)

    return X0_ptr, X1_ptr

# 乘法三元组
def Multiplication_triples(X, Y, workers):
    row_x, col_x = X.shape[0], X.shape[1]
    row_y, col_y = Y.shape[0], Y.shape[1]

    Delta_X = torch.randint(0, 2**n, (row_x, col_x))  # 随机矩阵
    Delta_Y_ = torch.randint(0, 2, (row_y, col_y))
    Delta_Z = torch.matmul(Delta_X, Delta_Y_)
    Delta_Y = Delta_Y_.type(torch.ByteTensor)

    Delta_X0_ptr, Delta_X1_ptr = Additive_A(Delta_X, workers)
    Delta_Y0_ptr, Delta_Y1_ptr = Additive_B(Delta_Y, workers)
    Delta_Z0_ptr, Delta_Z1_ptr = Additive_A(Delta_Z, workers)

    Delta_0_ptr = (Delta_X0_ptr, Delta_Y0_ptr, Delta_Z0_ptr)
    Delta_1_ptr = (Delta_X1_ptr, Delta_Y1_ptr, Delta_Z1_ptr)

    return Delta_0_ptr, Delta_1_ptr


def BitXA(X0_, Y0_, Z0_, delta_X0_, delta_Y0_, delta_Z0_,
          X1_, Y1_, Z1_, delta_X1_, delta_Y1_, delta_Z1_):

    assert X0_.shape[0] == Y0_.shape[0]
    assert X0_.shape[1] == Y0_.shape[1]

    row, col = X0_.shape[0], X0_.shape[1]
    # 通信交互，以打开x + delta_x, y, delta_y
    Delta_X0_, Delta_Y0_ = X0_ + delta_X0_, MatXor(Y0_, delta_Y0_)
    Delta_X0_copy, Delta_Y0_copy = Delta_Y0_.copy(), Delta_Y0_.copy()
    Delta_X0_bob, Delta_Y0_bob = Delta_X0_copy.move(bob), Delta_Y0_copy.move(bob)

    Delta_X1_, Delta_Y1_ = X1_ + delta_X1_, MatXor(Y1_, delta_Y1_)
    Delta_X1_copy, Delta_Y1_copy = Delta_X1_.copy(), Delta_Y1_.copy()
    Delta_X1_alice, Delta_Y1_alice = Delta_X1_copy.move(alice), Delta_Y1_copy.move(alice)

    Delta_X_alice, Delta_Y_alice = Delta_X0_ + Delta_X1_alice, MatXor(Delta_Y0_, Delta_Y1_alice)
    Delta_X_bob, Delta_Y_bob = Delta_X1_ + Delta_X0_bob, MatXor(Delta_Y1_, Delta_Y0_bob)

    Z0_ = (Delta_Y0_ * Delta_X0_ - 2 * Delta_Y_alice * Delta_X_alice * Delta_Y0_  
            - Delta_Y_alice * Delta_X0_ - delta_Z0_)

    Z1_ = (Delta_Y_bob * Delta_X_bob
            + Delta_Y1_ * Delta_X_bob - 2 * Delta_Y_bob * Delta_X_bob * Delta_Y1_
            - Delta_Y_bob * Delta_X1_ - delta_Z1_)


    # comm_total = sy.comm_total
    # for i in range(row):
    #     for j in range(col):
    #         Z0_[i,j] += (2 * Delta_Y_alice[i,j] * delta_Z0_[i,j])
    #         Z1_[i,j] += (2 * Delta_Y_bob[i,j] * delta_Z1_[i,j])

    # sy.comm_total = comm_total

    return Z0_, Z1_


if __name__=="__main__":

    X = torch.randint(0, 2**n, (N, N))
    Y_ = torch.randint(0, 2, (N, N))
    Y = Y_.type(torch.ByteTensor)
    Z = torch.matmul(X, Y_)

    Z_ = torch.zeros(N,N)
    Z0_, Z1_ = Additive_A(Z_, (alice, bob))

    # - - - - - Offline Phase - - - - - #
    X0_, X1_ = Additive_A(X, (alice, bob))
    Y0_, Y1_ = Additive_B(Y, (alice, bob))

    Delta_0_, Delta_1_ = Multiplication_triples(X, Y, (alice, bob))
    (Delta_X0_, Delta_Y0_, Delta_Z0_) = Delta_0_[0], Delta_0_[1], Delta_0_[2]
    (Delta_X1_, Delta_Y1_, Delta_Z1_) = Delta_1_[0], Delta_1_[1], Delta_1_[2]

    # - - - - - Online Phase - - - - - #

    sy.comm_total = 0           # commuication

    Z0_alice, Z1_bob = BitXA(X0_, Y0_, Z0_, Delta_X0_, Delta_Y0_, Delta_Z0_,
                             X1_, Y1_, Z1_, Delta_X1_, Delta_Y1_, Delta_Z1_,)

    Z_alice = Z0_alice.get()
    Z_bob = Z1_bob.get()

    comm_total = sy.comm_total

    print(('''bit length = {} bits, and dim of matrix is {} * {}\nTotal communication is {:.5f} MB'''.format(n, N, N, comm_total / 2**20)))
    

# bit length = 32 bits, and dim of matrix is 784 * 784
# Total communication is 5.27786 MB