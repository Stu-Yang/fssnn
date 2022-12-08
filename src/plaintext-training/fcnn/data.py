import torch
from torchvision import datasets, transforms


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
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        drop_last=True,
    )
    return train_loader, test_loader

