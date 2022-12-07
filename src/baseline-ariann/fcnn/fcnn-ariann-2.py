import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import syft as sy

from data import get_private_data_loaders
from procedure import train, test

# 定义一些常数
n_train_items = 1280
n_test_items = 1280

# 定义参与方Alice（P0）和Bob（P1），以及可信第三方crypto_provider
hook = sy.TorchHook(torch) 
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

workers = [alice, bob]
sy.local_worker.clients = workers


# 定义参数类
class Arguments():
    def __init__(self):
        self.batch_size = 128       # 训练时小批量大小
        self.test_batch_size = 32   # 验证时小批量大小

        self.n_train_items = n_train_items      # 调整训练数据条目数量
        self.n_test_items = n_test_items        # 调整测试数据条目数量

        self.epochs = 10            # 训练epoch大小
        self.lr = 0.01              # 学习率
        self.seed = 1
        self.momentum = 0.9
        self.log_interval = 1      # 每个epoch的日志信息
        self.precision_fractional = 3   # 小数部分的精度
        self.requires_grad = True       # requires_grad是Pytorch中通用数据结构Tensor的一个属性，用于说明当前量是否需要在计算中保留对应的梯度信息
        self.protocol = "fss"
        self.dtype = "long"

# 定义神经网络：使用3层全连接神经网络
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)     # 784 == 28*28
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)      # MNIST数据集的分类0～9，共10个类别
    
    def forward(self, x):                # 前向传播
        x = x.reshape(-1, 784)
        x = F.relu(self.fc1(x))          # 此处的relu函数是秘密协议中的relu函数
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

# 模型训练，并进行验证
if __name__ == "__main__":

    # 创建和定义参数
    args = Arguments()
    _ = torch.manual_seed(args.seed)    # 为CPU设置种子用于生成随机数

    encryption_kwargs = dict(      # 创建加密关键字参数
        workers=workers, crypto_provider=crypto_provider, protocol=args.protocol    # 在这里调用了fss
    )
    kwargs = dict(                  # 创建普通关键字参数
        requires_grad=args.requires_grad,   # requires_grad是Pytorch中通用数据结构Tensor的一个属性，用于说明当前量是否需要在计算中保留对应的梯度信息
        precision_fractional=args.precision_fractional,
        dtype=args.dtype,
        **encryption_kwargs,        # kwargs包含上述定义的加密关键字参数
    )

    # 打印模型训练信息
    print("================================================")
    print(f"Training over {args.epochs} epochs")
    print("model:\t\t", "Fully Connected Neural Network")
    print("dataset:\t", "MNIST")
    print("batch_size:\t", args.batch_size)
    print("================================================")

    # 获得密文状态下训练数据和测试数据
    private_train_loader, private_test_loader = get_private_data_loaders(args, kwargs)

    # 模型训练
    model = FCNN()
    model.encrypt(**kwargs)


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optimizer.fix_precision(
                    precision_fractional=args.precision_fractional, dtype=args.dtype
                )

    for epoch in range(1, args.epochs + 1):
        train(args, model, private_train_loader, optimizer, epoch)
    
    # 模型测试
    test(args, model, private_test_loader)