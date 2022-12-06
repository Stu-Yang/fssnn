import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# 定义参数，参考https://github.com/LaRiffle/ariann/blob/main/main.py中的默认参数
class Arguments():
    def __init__(self):
        self.batch_size = 128       # 训练时小批量大小
        self.test_batch_size = 32   # 验证时小批量大小
        self.epochs = 15            # 训练epoch大小
        self.lr = 0.01              # 学习率
        self.momentum = 0.9
        self.log_interval = 50

args = Arguments()

# 自定义ReLU函数
def relu(x):
    x[x < 0] = 0
    return x

# 定义神经网络：使用3层全连接神经网络，参考https://github.com/LaRiffle/ariann/blob/main/models.py中Network1
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 784 == 28*28
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)    # MNIST数据集的分类0～9，共10个类别
    
    def forward(self, x):               # 前向传播
        x = x.reshape(-1, 784)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        return x

# 训练数据和测试数据加载，参考https://github.com/LaRiffle/ariann/blob/main/data.py
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, drop_last=True)

# 定义one-hot编码函数
def one_hot_of(index_tensor):
    """
    Transform to one hot tensor
    
    Example:
        [0, 3, 9]
        =>
        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]

    """
    onehot_tensor = torch.zeros(*index_tensor.shape, 10)                    # 10 classes for MNIST
    onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
    return onehot_tensor

# 自定义均方误差函数
def mse_loss_mean(output, target):
    batch_size = output.shape[0]
    loss = ((output - target) ** 2).sum() / batch_size
    return loss

def mse_loss_sum(output, target):
    loss = ((output - target) ** 2).sum()
    return loss

# 定义训练过程，参考https://github.com/LaRiffle/ariann/blob/main/procedure.py
def train(args, model, train_loader, optimizer, epoch):
    model.train()       # 设置为trainning模式
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()

        optimizer.zero_grad()
        output = model(data)        # 前向传播计算
        loss = mse_loss_mean(output, one_hot_of(target))       # 计算误差，其中output是batch size * class size的矩阵，target是batch size的向量

        loss.backward()          # 反向传播
        optimizer.step()         # 更新参数
        total_time = time.time() - start_time

        if batch_idx % args.log_interval == 0:      # 打印信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
                epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size,
                100.0 * batch_idx / len(train_loader), loss.item(), total_time))

# 定义模型测试过程，参考https://github.com/LaRiffle/ariann/blob/main/procedure.py
def test(args, model, test_loader):
    model.eval()        # 设置为test模式  
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += mse_loss_sum(output, one_hot_of(target)).item()     # sum up batch loss
            pred = output.argmax(1, keepdim=True)       # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 模型训练，并进行验证
if __name__ == "__main__":

    # 模型训练
    print("---------- Training ----------")
    model = FCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)

    training_time = time.time() - start_time
    print("\nOnline Training Time: {:.3f}s ".format(training_time))


    # 模型验证
    test(args, model, test_loader)