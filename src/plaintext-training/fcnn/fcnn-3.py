# 下面考虑3层全连接神经网络（Fully Connected Nerual Network，FCNN），数据集为MNIST
# 相比于fcnn-2.py的修改为：
#   1. 使用的ReLU函数为自定义函数
#   2. 根据https://github.com/LaRiffle/ariann重新设置了参数类Arguments()中的参数
#   3. 使用的优化器新增了参数momentum
#   4. 增加训练时间衡量指标

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 定义参数（batch大小、学习率等）
class Arguments():
    def __init__(self):
        self.batch_size = 128       # 训练时小批量大小
        self.test_batch_size = 32   # 验证时小批量大小
        self.epochs = 15            # 训练epoch大小
        self.lr = 0.01              # 学习率
        self.momentum = 0.9
        self.log_interval = 50

args = Arguments()

# 训练数据和测试数据加载
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

# 自定义ReLU函数
def relu(x):
    x[x < 0] = 0
    return x

# 定义神经网络：使用3层全连接神经网络(输入层，隐含层和输出层；1FC-1ReLU-1FC-1ReLU-1FC)
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fclayer1 = nn.Linear(28*28, 500)
        #self.relu = nn.ReLU()
        self.fclayer2 = nn.Linear(500, 500)
        #self.relu = nn.ReLU()
        self.fclayer3 = nn.Linear(500, 10)
    
    def forward(self, x):                # 前向传播
        x = x.view(-1, 28*28)
        x = self.fclayer1(x)
        x = relu(x)
        x = self.fclayer2(x)
        x = relu(x)
        x = self.fclayer3(x)
        # x = relu(x)
        return x

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

# 定义训练过程
def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()

        optimizer.zero_grad()
        output = model(data)                    # 前向传播
        output = F.log_softmax(output, dim=1)   # 对输出进行归一化（按列进行）
        loss = F.mse_loss(output, one_hot_of(target))       # 计算误差，其中output是batch size * class size的矩阵，target是batch size的向量
        loss.backward()                         # 反向传播
        optimizer.step()                        # 更新参数

        total_time = time.time() - start_time

        if batch_idx % args.log_interval == 0:      # 打印信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
                epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size,
                100.0 * batch_idx / len(train_loader), loss.item(), total_time))

# 模型训练过程
print("---------- Training ----------")
model = FCNN()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

start_time = time.time()
for epoch in range(1, args.epochs + 1):
    train(args, model, train_loader, optimizer, epoch)

training_time = time.time() - start_time
print("\n---------- Training Time: {:.3f}s ----------".format(training_time))


# 模型测试过程
def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.mse_loss(output, one_hot_of(target)).item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

print("\n---------- Testing ----------")
test(args, model, test_loader)