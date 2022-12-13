import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import syft as sy
from syft.serde.compression import NO_COMPRESSION
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

from data import get_data_loaders       # 读取训练数据集和测试数据集
from procedure import train, test       # 定义训练函数和测试函数

# 定义参数，参考https://github.com/LaRiffle/ariann/blob/main/main.py中的默认参数
class Arguments():
    def __init__(self):
        self.batch_size = 128       # 训练时小批量大小
        self.test_batch_size = 32   # 验证时小批量大小
        self.epochs = 15            # 训练epoch大小
        self.lr = 0.01              # 学习率
        self.momentum = 0.9
        self.log_interval = 50      # 日志信息打印间隔长度
        self.precision_fractional = 5   # 小数部分的精度
        self.requires_grad = True       # requires_grad是Pytorch中通用数据结构Tensor的一个属性，用于说明当前量是否需要在计算中保留对应的梯度信息
        self.protocol = "fss"
        self.dtype = "long"
        self.n_train_items = -1     # 表示使用全部训练数据
        self.n_test_items = -1      # 表示使用全部测试数据


# 定义神经网络：使用3层全连接神经网络，参考https://github.com/LaRiffle/ariann/blob/main/models.py中Network1
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 784 == 28*28
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)    # MNIST数据集的分类0～9，共10个类别
    
    def forward(self, x):               # 前向传播
        x = x.reshape(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


# 模型训练，并进行测试
if __name__ == "__main__":

    # 创建参数
    args = Arguments()

    # 打印模型训练信息
    print("================================================")
    print(f"(Fix Precision) Training over {args.epochs} epochs")
    print("model:\t\t", "Fully Connected Neural Network")
    print("dataset:\t", "MNIST")
    print("batch_size:\t", args.batch_size)
    print("================================================")

    # 创建工作机
    hook = sy.TorchHook(torch)
    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    workers = [alice, bob]
    sy.local_worker.clients = workers

    # 定义参数
    encryption_kwargs = dict(
        workers=workers, crypto_provider=crypto_provider, protocol=args.protocol
    )
    kwargs = dict(
        requires_grad=args.requires_grad,
        precision_fractional=args.precision_fractional,
        dtype=args.dtype,
        **encryption_kwargs,
    )

    # 获取数据（明文状态下训练数据和测试数据）
    train_loader, test_loader = get_data_loaders(args, kwargs)

    # 模型训练
    model = FCNN()

    model.encrypt(**kwargs)     # .encrypt()方法定义在/root/gsq_workplace/arinn/PySyft/syft/frameworks/torch/hook/hook.py(738)
    model.get()                 # .get()方法定义在/root/gsq_workplace/arinn/PySyft/syft/frameworks/torch/hook/hook.py(721)

    for epoch in range(args.epochs):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        optimizer = optimizer.fix_precision(    # .fix_precision方法定义在/root/gsq_workplace/arinn/PySyft/syft/frameworks/torch/hook/hook.py(929)
                precision_fractional=args.precision_fractional, dtype=args.dtype
            )
            
        train_time = train(args, model, train_loader, optimizer, epoch)
        test_time, accuracy = test(args, model, test_loader)