import torch
from torchvision import datasets, transforms

class Arguments():
    def __init__(self):
        self.batch_size = 128       # 训练时小批量大小
        self.test_batch_size = 32   # 验证时小批量大小
        self.epochs = 15            # 训练epoch大小
        self.lr = 0.01              # 学习率
        self.momentum = 0.9
        self.log_interval = 50      # 日志信息打印间隔长度


# 数据集加载函数
def get_data_loaders(args):

    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transformation
    )
    test_dataset = datasets.MNIST(
        "../data", train=False, download=True, transform=transformation
    )

    train_loader = torch.utils.data.DataLoader(      # torch.utils.data.DataLoader()函数是PyTorch中数据读取的接口
        train_dataset,                  # 加载数据的数据集，由上述dataset取值来确定
        batch_size=args.batch_size,     # 每个batch加载多少个样本
        drop_last=True,                 # 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch
    )
    
    new_train_loader = []
    for (data, target) in enumerate(train_loader):
        new_train_loader.append((data, (target)))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        drop_last=True,
    )

    new_test_loader = []
    for data, target in enumerate(test_loader):
        new_test_loader.append((data, target))


    return new_train_loader, new_test_loader


# 创建参数
args = Arguments()

# 获取数据（明文状态下训练数据和测试数据）
public_train_loader, public_test_loader = get_data_loaders(args)

n_train_items = (len(public_train_loader) - 1) * args.batch_size + len(
            public_train_loader[-1][1]
        )

# print("public_train_loader_len = ", len(public_train_loader.dataset),
#       "public_test_loader_len  = ", len(public_test_loader.dataset))

print("n_train_items = ", n_train_items)