import time
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# 自定义ReLU函数
def relu(x):
    x[x < 0] = 0
    return x

# 定义神经网络：使用3层全连接神经网络，参考https://github.com/LaRiffle/ariann/blob/main/models.py中Network1
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 784 == 28*28
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)    # MNIST数据集的分类0～9，共10个类别
    
    def forward(self, x):               # 前向传播
        x = x.reshape(-1, 784)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        return x


# 模型训练，并进行测试
if __name__ == "__main__":

    # 创建参数
    args = Arguments()

    # 打印模型训练信息
    print("================================================")
    print(f"Plaintext Training over {args.epochs} epochs")
    print("model:\t\t", "Fully Connected Neural Network")
    print("dataset:\t", "MNIST")
    print("batch_size:\t", args.batch_size)
    print("================================================")

    # 获取数据（明文状态下训练数据和测试数据）
    public_train_loader, public_test_loader = get_data_loaders(args)    # len(train)=60000, len(test)=10000

    # 模型训练
    model = FCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)       # 优化器不使用momentum时，训练精度会大大降低
    
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, public_train_loader, optimizer, epoch)
        print()
    training_time = time.time() - start_time


    print("============================================================")
    print("Online Training Time: {:.3f}s ".format(training_time))

    # 模型测试
    test(args, model, public_test_loader)
    print("============================================================")