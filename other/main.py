'''
项目：局域网设置进行隐私保护神经网络模型训练，获得密文状态下的训练时间、训练精度和通信量
模型：三层全连接前馈神经网络模型
数据集：MNIST
损失函数：均方误差函数
激活函数：ReLU函数，f(x) = max(0,x)
依赖：Python 3.7.0, Syft 0.2.9, torch 1.4.0+cpu, torchvision 0.5.0
'''

import argparse    # argparse是python中的命令行选项、参数和子命令解析器

import torch
import torch.optim as optim
import syft as sy
hook = sy.TorchHook(torch)   # 创建钩子

from procedure import train, test
from data import get_data_loaders, get_number_classes
from models import get_model, load_state_dict

def run(args):
    # 创建工作节点
    bob = sy.VirtualWorker(hook, id="bob")                              #创建工作节点，命名为bob（参与方P1）
    alice = sy.VirtualWorker(hook, id="alice")                          #创建工作节点，命名为alice（参与方P0）
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")      #创建工作节点，命名为crypto_provider（第三方），其可以提供随机数

    workers = [alice, bob]
    sy.local_worker.clients = workers

    encryption_kwargs = dict(      # 创建加密关键字参数
        workers=workers, crypto_provider=crypto_provider, protocol=args.protocol    # 在这里调用了fss，因为此处的args.protocol在类对象args中被定义为fss！！！！！！！！！！
    )
    kwargs = dict(                  # 创建普通关键字参数
        requires_grad=args.requires_grad,   # requires_grad是Pytorch中通用数据结构Tensor的一个属性，用于说明当前量是否需要在计算中保留对应的梯度信息
        precision_fractional=args.precision_fractional,
        dtype=args.dtype,
        **encryption_kwargs,        # kwargs包含上述定义的加密关键字参数
    )

    private_train_loader, private_test_loader = get_data_loaders(args, kwargs, private=True)    # 获取数据（密文状态下训练数据和测试数据），get_data_loaders()函数被定义在./data.py文件中
    public_train_loader, public_test_loader = get_data_loaders(args, kwargs, private=False)     # 获取数据（明文状态下训练数据和测试数据）
    model = get_model(args.model, args.dataset, out_features=get_number_classes(args.dataset))  # 获取模型，get_model()在model.py中定义

    if args.test and not args.train:
        load_state_dict(model, args.model, args.dataset)      # load_state_dict函数被定义在./model.py中，其作用是将预训练模型的参数权重加载到模型model中

    model.encrypt(**kwargs)


    for epoch in range(args.epochs):
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)   # 实现随机梯度下降（SGD）算法，参见https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/
        optimizer = optimizer.fix_precision(
                    precision_fractional=args.precision_fractional, dtype=args.dtype
                )
        
        train_time = train(args, model, private_train_loader, optimizer, epoch)
    
    print("**********Training is Done!**********")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()   # 创建一个新的 ArgumentParser 对象，该对象结合add_argument(*args)使用，表示执行时需要*args（可参考https://docs.python.org/zh-cn/3/howto/argparse.html#id1）

    parser.add_argument(    # 选取神经网络模型，另外add_argument()方法用于指定程序能够接受哪些命令行选项。（可参考https://docs.python.org/zh-cn/3/howto/argparse.html#id1）
        "--model",          # 命令行参数model，其中--表示该参数时可选参数
        type=str,           # 在这里规定了该参数的类型，默认情况下时字符串类型
        help="model to use for inference (network1, network2, lenet, alexnet, vgg16, resnet18)",   # 这里添加了一些显示信息
    )

    parser.add_argument(    # 选取数据集
        "--dataset",
        type=str,
        help="dataset to use (mnist, cifar10, hymenoptera, tiny-imagenet)",
    )

    parser.add_argument(    # 选取小批量大小，训练过程中默认为128
        "--batch_size",
        type=int,
        help="size of the batch to use. Default 128.",
        default=128,        # 规定该参数的默认值
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        help="size of the batch to use",
        default=None,
    )

    parser.add_argument(    # 注明是否是测试
        "--test",
        help="run testing on the complete test dataset",
        action="store_true",
    )

    parser.add_argument(    # 注明是否是训练
        "--train",
        help="run training for n epochs",
        action="store_true",
    )

    parser.add_argument(    # 选取epoch大小，训练过程中默认是15
        "--epochs",
        type=int,
        help="[needs --train] number of epochs to train on. Default 15.",
        default=15,
    )

    parser.add_argument(    # SGD算法学习率，默认是0.01
        "--lr",
        type=float,
        help="[needs --train] learning rate of the SGD. Default 0.01.",
        default=0.01,
    )

    parser.add_argument(    # 是否采用动量SGD算法进行训练
        "--momentum",
        type=float,
        help="[needs --train] momentum of the SGD. Default 0.9.",
        default=0.9,
    )

    cmd_args = parser.parse_args()   # 把parser中设置的所有"add_argument"给返回到cmd_args子类实例当中（那么parser中增加的参数都会在args实例中），可以通过cmd_args.xxx来调用

    class Arguments:     # 新建参数类
        model = cmd_args.model.lower()  # cmd_args.model是str类型，.lower()是将str全部转换为小写
        dataset = cmd_args.dataset.lower()  # cmd_args.dataset是str类型，.lower()是将str全部转换为小写

        train = cmd_args.train
        n_train_items = -1 if cmd_args.train else cmd_args.batch_size
        test = cmd_args.test or cmd_args.train
        n_test_items = -1 if cmd_args.test or cmd_args.train else cmd_args.batch_size

        batch_size = cmd_args.batch_size
        # Defaults to the train batch_size
        test_batch_size = cmd_args.test_batch_size or cmd_args.batch_size

        epochs = cmd_args.epochs
        lr = cmd_args.lr
        momentum = cmd_args.momentum
        requires_grad = cmd_args.train
        dtype = "long"

    args = Arguments()     # 新建参数类对象args

    run(args)          # 直接运行上述参数args（run()函数在开头进行定义），其是一个Arguments类（在本文件定义），该类中的对象由parser的子类示例来赋值





