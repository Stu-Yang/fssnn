import torch
from torchvision import datasets, transforms

# 本文件主要参考https://github.com/Stu-Yang/ariann/blob/main/data.py

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

# 定义训练数据和测试数据加载函数
def get_private_data_loaders(args, kwargs):
    
    def encode(tensor):     # 对数据进行加密
        encrypted_tensor = tensor.encrypt(**kwargs)     # 在这里，tensor是调用encode()的样本数据data或样本标签target，encrypt方法�?root/gsq_workplace/arinn/PySyft/syft/frameworks/torch/tensors/interpreters/native.py中定义
        return encrypted_tensor
    
    # 在这里，使用mnist数据集   
    transformation = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transformation),
        batch_size=args.batch_size,
        num_workers = 4
    )
    
    private_train_loader = []
    for i, (data, target) in enumerate(train_loader):    # data是样本数据，target是样本标�?
        if args.n_train_items >= 0 and i >= args.n_train_items / args.batch_size:
            break
        private_train_loader.append((encode(data), encode(one_hot_of(target))))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transformation),
        batch_size=args.test_batch_size,
        num_workers = 4
    )
    
    private_test_loader = []
    for i, (data, target) in enumerate(test_loader):
        if args.n_test_items >= 0 and i >= args.n_test_items / args.test_batch_size:
            break
        private_test_loader.append((encode(data), encode(target.float())))
    
    return private_train_loader, private_test_loader