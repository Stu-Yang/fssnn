import time
import torch
import syft as sy

# 本文件主要参考自https://github.com/LaRiffle/ariann/blob/main/procedure.py

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
    onehot_tensor = torch.zeros(*index_tensor.shape, 10)  # 10 classes for MNIST
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

    test_loss /= (len(test_loader) * args.test_batch_size)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, (len(test_loader) * args.test_batch_size),
        100. * correct / (len(test_loader) * args.test_batch_size)))
