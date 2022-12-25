import torch
import syft as sy

# 创建工作机
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")


def matxor(X, Y):
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
    X0, X1 = matxor(X, R), R
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




n = 16      # bit length
N = 784     # matrix dim

X = torch.randint(0, 2**n, (N, N))
Y_ = torch.randint(0, 2, (N, 1))
Y = Y_.type(torch.ByteTensor)
Z = torch.matmul(X, Y_)


# - - - - - Offline Phase - - - - - #
X0_, X1_ = Additive_A(X, (alice, bob))
Y0_, Y1_ = Additive_A(Y, (alice, bob))

delta_0_, delta_1_ = Multiplication_triples(X, Y, (alice, bob))
(delta_X0_, delta_Y0_, delta_Z0_) = delta_0_[0], delta_0_[1], delta_0_[2]
(delta_X1_, delta_Y1_, delta_Z1_) = delta_1_[0], delta_1_[1], delta_1_[2]


# - - - - - Online Phase - - - - - #
sy.comm_total = 0           # commuication

# 打开X + delta_X, Y, delta_Y
Delta_X0_, Delta_Y0_ = X0_ + delta_X0_, matxor(Y0_, delta_Y0_)
Delta_X0_bob, Delta_Y0_bob = Delta_X0_.copy(), Delta_Y0_.copy()
Delta_X0_bob, Delta_Y0_bob = Delta_X0_bob.move(bob), Delta_Y0_bob.move(bob)

Delta_X1_, Delta_Y1_ = X1_ + delta_X1_, matxor(Y1_, delta_Y1_)
Delta_X1_alice, Delta_Y1_alice = Delta_X1_.copy(), Delta_Y1_.copy()
Delta_X1_alice, Delta_Y1_alice = Delta_X1_alice.move(alice), Delta_Y1_alice.move(alice)

Delta_X_alice, Delta_Y_alice = Delta_X0_ + Delta_X1_alice, matxor(Delta_Y0_, Delta_Y1_alice)
Delta_X_bob, Delta_Y_bob = Delta_X1_ + Delta_X0_bob, matxor(Delta_Y1_, Delta_Y0_bob)


Delta_X_alice = Delta_X_alice.type(torch.IntTensor)
delta_Y0_ = delta_Y0_.type(torch.IntTensor)
Delta_Y_alice = Delta_Y_alice.type(torch.IntTensor)

Delta_X_bob = Delta_X_bob.type(torch.IntTensor)
delta_Y1_ = delta_Y1_.type(torch.IntTensor)
Delta_Y_bob = Delta_Y_bob.type(torch.IntTensor)

Z0_ = (torch.matmul(Delta_X_alice, delta_Y0_) 
        - torch.matmul(torch.matmul(Delta_X_alice, Delta_Y_alice), delta_Y0_) 
        - torch.matmul(Delta_Y_alice, delta_Y0_) 
        - delta_Z0_ 
        + 2 * torch.matmul(Delta_Y_alice, delta_Z0_))

Z1_ = (torch.matmul(Delta_X_bob, Delta_Y_bob) 
        + torch.matmul(delta_Y1_, Delta_X_bob) 
        - 2 * torch.matmul(torch.matmul(Delta_X_bob, Delta_Y_bob), delta_Y1_) 
        - torch.matmul(Delta_Y_bob, delta_Y1_) 
        - delta_Z1_ 
        + 2 * torch.matmul(Delta_Y_bob, delta_Z1_))



comm_total = sy.comm_total
print("bit length = {} bits, and dim of matrix is {} * {}\n Total communication is {:.5f} MB".format(n, N, N, comm_total / 2**20))

