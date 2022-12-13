import torch
from torchvision import datasets, transforms

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


# 数据集加载函数
def get_data_loaders(args, kwargs):
    def encode(tensor):
        """
        Depending on the setting, acts on a tensor
        - Do nothing
        OR
        - Transform to fixed precision
        OR
        - Secret share
        """
        encrypted_tensor = tensor.encrypt(**kwargs)     # 定义在/root/gsq_workplace/arinn/PySyft/syft/frameworks/torch/tensors/interpreters/native.py(1099)
        return encrypted_tensor.get()

    transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transformation
    )
    test_dataset = datasets.MNIST(
        "../data", train=False, download=True, transform=transformation
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
    )

    new_train_loader = []
    for i, (data, target) in enumerate(train_loader):
        if args.n_train_items >= 0 and i >= args.n_train_items / args.batch_size:
            break
        new_train_loader.append((encode(data), encode(one_hot_of(target))))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        drop_last=True,
    )

    new_test_loader = []
    for i, (data, target) in enumerate(test_loader):
        if args.n_test_items >= 0 and i >= args.n_test_items / args.test_batch_size:
            break
        new_test_loader.append((encode(data), encode(target.float())))

    return new_train_loader, new_test_loader

