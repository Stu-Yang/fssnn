import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import syft as sy

# 定义参与方Alice（P0）和Bob（P1），以及可信第三方crypto_provider
hook = sy.TorchHook(torch) 
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider") 
workers = [alice, bob]

# 定义一些常数
n_train_items = 320
n_test_items = 320

# 定义参数
class Arguments():
    def __init__(self):
        self.batch_size = 128       # 训练时小批量大小
        self.test_batch_size = 32   # 验证时小批量大小
        self.epochs = 2            # 训练epoch大小
        self.lr = 0.01              # 学习率
        self.seed = 1
        self.momentum = 0.9
        self.log_interval = 1      # 每个epoch的日志信息
        self.precision_fractional = 3   # 小数部分的精度

args = Arguments()
_ = torch.manual_seed(args.seed)    # 为CPU设置种子用于生成随机数

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


# 定义训练数据和测试数据加载函数
def get_private_data_loaders(precision_fractional, workers, crypto_provider):
    
    def one_hot_of(index_tensor):   # one hot编码函数
        """
        Transform to one hot tensor
        
        Example:
            [0, 3, 9]
            =>
            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
            
        """
        onehot_tensor = torch.zeros(*index_tensor.shape, 10) # 10 classes for MNIST
        onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
        return onehot_tensor
        
    def secret_share(tensor):       # 秘密分享函数
        """
        Transform to fixed precision and secret share a tensor
        """
        return (
            tensor
            .fix_precision(precision_fractional=precision_fractional)
            .share(*workers, crypto_provider=crypto_provider, protocol="fss", requires_grad=True)
        )
    
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transformation),
        batch_size=args.batch_size
    )
    
    private_train_loader = [
        (secret_share(data), secret_share(one_hot_of(target)))
        for i, (data, target) in enumerate(train_loader)
        if i < n_train_items / args.batch_size
    ]
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transformation),
        batch_size=args.test_batch_size
    )
    
    private_test_loader = [
        (secret_share(data), secret_share(target.float()))
        for i, (data, target) in enumerate(test_loader)
        if i < n_test_items / args.test_batch_size
    ]
    
    return private_train_loader, private_test_loader
    
    
private_train_loader, private_test_loader = get_private_data_loaders(
    precision_fractional=args.precision_fractional,
    workers=workers,
    crypto_provider=crypto_provider
)


# 自定义均方误差函数
def mse_loss_mean(output, target):
    batch_size = output.shape[0]
    loss = ((output - target) ** 2).sum().refresh() / batch_size    # Refresh shares by adding shares of zero
    return loss

def mse_loss_sum(output, target):
    loss = ((output - target) ** 2).sum()
    return loss

# 定义训练过程
def train(args, model, private_train_loader, optimizer, epoch):
    model.train()       # 设置为trainning模式
    for batch_idx, (data, target) in enumerate(private_train_loader):   # 私有数据集
        start_time = time.time()

        optimizer.zero_grad()
        output = model(data)        # 前向传播计算
        loss = mse_loss_mean(output, target)       # 计算误差，其中output是batch size * class size的矩阵，target是batch size的向量

        loss.backward()          # 反向传播
        optimizer.step()         # 更新参数
        total_time = time.time() - start_time

        if batch_idx % args.log_interval == 0:      # 打印信息
            loss = loss.get().float_precision()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
                epoch, batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
                100. * batch_idx / len(private_train_loader), loss.item(), total_time))

# 模型验证过程
# def test(args, model, private_test_loader):
#     model.eval()        # 设置为test模式  
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in private_test_loader:
#             output = model(data)
#             test_loss += mse_loss_sum(output, target).item()     # sum up batch loss
#             pred = output.argmax(1, keepdim=True)       # get the index of the max log-probability 
#             correct += pred.eq(target.view_as(pred)).sum().item()
    
#     correct = correct.get().float_precision()
#     test_loss = test_loss.get().float_precision()
#     test_loss /= len(private_test_loader.dataset)

#     print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
#         test_loss, correct, len(private_test_loader.dataset),
#         100. * correct / len(private_test_loader.dataset)))

def test(args, model, private_test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in private_test_loader:
            
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()

    correct = correct.get().float_precision()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct.item(), len(private_test_loader)* args.test_batch_size,
        100. * correct.item() / (len(private_test_loader) * args.test_batch_size)))


# 模型训练，并进行验证
if __name__ == "__main__":

    # 模型训练
    model = FCNN()
    model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, protocol="fss", requires_grad=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optimizer.fix_precision() 

    for epoch in range(1, args.epochs + 1):
        train(args, model, private_train_loader, optimizer, epoch)
    
    # 模型验证
    test(args, model, private_test_loader)