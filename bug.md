## 实验问题记录

### 0. 背景介绍

目前，我们完成了论文的理论部分，正在进行实验验证。我们的实验是基于参照论文[AriaNN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing](https://petsymposium.org/popets/2022/popets-2022-0015.php)的源码[GitHub-ariann](https://github.com/LaRiffle/ariann)进行修改而来，目前我们已经跑通了源码[GitHub-ariann](https://github.com/LaRiffle/ariann)，并且测试了相关的性能。

源码[GitHub-ariann](https://github.com/LaRiffle/ariann)是基于[Pysyft](https://github.com/OpenMined/PySyft)构建的，PySyft是一个提供安全计算的Python开源框架，支持联邦学习、安全多方计算等隐私计算技术。源码[GitHub-ariann](https://github.com/LaRiffle/ariann)使用的Pysyft版本为[PySyft_0.2.x](https://github.com/OpenMined/PySyft/tree/PySyft/syft_0.2.x)，该版本已经不再维护，而且不被后续版本兼容（后续PySyft_0.3+全面修改了Python接口）。尽管如此，[PySyft_0.2.x](https://github.com/OpenMined/PySyft/tree/PySyft/syft_0.2.x)依然能正常工作。

为了更好地进行代码修改，我们只保留了源码[GitHub-ariann](https://github.com/LaRiffle/ariann)中所被需要的部分（机器学习训练部分），并形成了我们的源码，该源码可以正常运行，并且可以达到源码[GitHub-ariann](https://github.com/LaRiffle/ariann)的性能。

### 1. 如何复现？

#### 1.1 远程连接服务器

如果能远程连接服务器，可以在VsCode远程资源管理器SSH Targets的配置文件中添加：
```
Host CARC_yp
  HostName 219.223.251.135
  Port 18102
  User root
```
添加后，进行远程连接，所需要源代码在下面所述的文件夹中
```
|-- __pycache__
|-- data.py             # 数据文件，负责产生训练数据和测试数据
|-- fcnn-ariann-3.py    # 主文件
|-- fcnn-ariann-3_threading.py  # 主文件（添加多进程版本）
|-- procedure.py                # 训练和测试文件
`-- procedure_threading.py      # 训练和测试文件（添加多进程版本）

```

然后，执行下面的命令即可：
```
(base) root@yangpeng:~# conda activate pysyft           # 激活python虚拟环境pysyft
(pysyft) root@yangpeng:~# cd /root/yp_workplace/fssnn   # 进入对应文件夹
(pysyft) root@yangpeng:~/yp_workplace/fssnn# python fcnn-ariann-3.py    # 运行源代码文件
================================================
(AriaNN) Ciphertext Training over 15 epochs
model:           Fully Connected Neural Network
dataset:         MNIST
batch_size:      128
================================================
......
```

#### 1.2 源码部署

首先打包下载[code-download](https://resource-1305526482.cos.ap-guangzhou.myqcloud.com/Code/fssnn.zip)，然后解压到fssnn文件夹，并执行以下命令:

```
>>> conda create -n pysyft python=3.7       # 创建python虚拟环境
>>> conda activate pysyft                   # 激活环境pysyft
>>> cd /root/fssnn/PySyft                   # 进入PySyft文件夹
>>> pip install -e .                        # 部署PySyft
>>> cd /root/fssnn/fssnn                    # 进入fssnn文件夹
>>> python fcnn-ariann-3.py                 # 运行源代码文件 
```

部署过程中可能会缺少一些依赖而报错，我所用到的一些依赖如下：

<details>
<summary>下面是PySyft安装所需要用到的一些依赖，请点击展开:mag:</summary>

```
Package              Version     Editable project location
-------------------- ----------- --------------------------------
aioice               0.6.18
aiortc               0.9.28
attrs                22.1.0
av                   8.1.0
backcall             0.2.0
beautifulsoup4       4.11.1
bidict               0.22.0
bleach               5.0.1
certifi              2022.9.24
cffi                 1.15.1
chardet              3.0.4
click                7.1.2
crc32c               2.3
cryptography         38.0.4
debugpy              1.6.4
decorator            5.1.1
defusedxml           0.7.1
dill                 0.3.6
entrypoints          0.4
fastjsonschema       2.16.2
Flask                1.1.4
Flask-SocketIO       4.2.1
idna                 2.8
importlib-metadata   5.1.0
importlib-resources  1.5.0
ipykernel            6.9.2
ipython              7.34.0
ipython-genutils     0.2.0
itsdangerous         1.1.0
jedi                 0.18.2
Jinja2               2.11.3
jsonschema           4.17.3
jupyter-client       7.1.2
jupyter_core         4.12.0
jupyterlab-pygments  0.2.2
lz4                  3.0.2
MarkupSafe           2.1.1
matplotlib-inline    0.1.6
mistune              0.8.4
msgpack              1.0.4
msgpack-numpy        0.4.8
nbclient             0.5.13
nbconvert            6.4.5
nbformat             5.7.0
nest-asyncio         1.5.6
netifaces            0.11.0
notebook             5.7.8
numpy                1.18.5
openmined.threepio   0.2.0
pandocfilters        1.5.0
parso                0.8.3
pexpect              4.8.0
phe                  1.4.0
pickleshare          0.7.5
Pillow               9.3.0
pip                  22.2.2
pkgutil_resolve_name 1.3.10
prometheus-client    0.15.0
prompt-toolkit       3.0.36
protobuf             3.19.0
psutil               5.7.0
ptyprocess           0.7.0
pyarrow              2.0.0
pycparser            2.21
pyee                 9.0.4
Pygments             2.13.0
pylibsrtp            0.7.1
pyrsistent           0.19.2
python-dateutil      2.8.2
python-engineio      4.3.4
python-socketio      5.7.2
pyzmq                24.0.1
requests             2.22.0
requests-toolbelt    0.9.1
RestrictedPython     5.2
scipy                1.4.1
Send2Trash           1.8.0
setuptools           63.4.1
six                  1.16.0
soupsieve            2.3.2.post1
sycret               0.1.3
syft                 0.2.9       /root/yp_workplace/ariann/PySyft
syft-proto           0.5.3
tblib                1.6.0
terminado            0.13.3
testpath             0.6.0
torch                1.4.0
torchvision          0.5.0
tornado              4.5.3
traitlets            5.7.1
typing_extensions    4.4.0
urllib3              1.25.11
wcwidth              0.2.5
webencodings         0.5.1
websocket-client     0.57.0
websockets           8.1
Werkzeug             1.0.1
wheel                0.37.1
zipp                 3.11.0
```
</details>

### 2. 实验问题

### 2.0 问题记录

在密文状态下固定精度全连接神经网络训练（fcnn-ariann）中我们遇到了一些问题，我们的实验设置如下：
+ **全连接神经网络模型训练**
  +  数据集：MNIST数据集（60000个训练集样本，10000个测试集样本）
  +  神经网络模型：三层全连接前馈神经网络（1FC(784 X 128)-1ReLU-1FC(128 X 128)-1ReLU-1FC(128 X 10)-1ReLU）
  +  模型参数
    + batch数：训练时batch大小为128，测试时batch大小为32
    + epoch数：15
    + 学习率：0.01
    + 小数部分位数：5位（部分实验采用了3位）
  + 优化器：SGD优化器，采用momentum模式，取值0.9
  + 损失函数：均方误差函数
  + 激活函数：ReLU函数
  + **加密协议：`fss`**（函数秘密共享） 
+ 实验结果记录
  + 第一次实验
    + `TEST Accuracy: 9758.0/9984 (97.74%) 	Time /item: 0.6398s 	Time w. argmax /item: 0.6500s [32.000]`
    + `Online Training Time: 383988.130s(106.66h)`，这是15个epoch的训练时间，计算可得每个epoch需要7.11h

训练精度是合理的，但是训练时间是不合理的，根据我们的参照论文[AriaNN](https://petsymposium.org/popets/2022/popets-2022-0015.php)中Table的第一行数据可知，每个epoch所需时间应该在0.78h，这和我们的实验相差很大。通过直接运行源码[GitHub-ariann](https://github.com/LaRiffle/ariann)我们得到的结果也是每个epoch所需时间7h左右，因此我们排除了代码本身的问题，转而认为是服务器配置的差距。

![image](https://user-images.githubusercontent.com/66773755/208018770-b1300bb3-f9c0-42fc-adc8-c2c7d29dd2ec.png)


注：其中的network-1即为我们设置的全连接神经网络，论文中的所有配置都和我们的设置相同。

#### 2.1 实验的尝试

因此，我们的思路是利用cpu多核多线程来运行程序，主要的尝试如下：
+ 首先，我们查看了服务器配置（该服务器即为1.1 远程连接服务器中所提到的服务器），以及运行`python fcnn-ariann-3.py`时的进程状况
    + 服务器配置：cpu信息：`32  Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz`，物理cpu个数：`2`，cpu核心数：`8`，逻辑cpu个数 (物理cpu个数 * cpu核心数 * 超线程数)：`32`
    + `python fcnn-ariann-3.py`进程对应的CPU利用率大概为`100% ~ 1150%`
+ 其次，考虑到我们使用的框架是基于PySyft框架的，而该框架是基于PyTorch的，因此试图利用`torch.set_num_threads(xx)`来控制进程执行情况，但发现该命令没有明显的作用
+ 最后，我们试图基于[pytorch/examples](https://github.com/pytorch/examples/tree/main/mnist_hogwild)将我们的源代码`fcnn-ariann-3.py`和`procedure.py`转换为对应的多进程版本`fcnn-ariann-3_threading.py`和`procedure_threading.py`，但一直会报错，其主要原因我觉得是我们的代码文件`fcnn-ariann-3.py`和`procedure.py`中使用了加密等操作，因此在多进程版本中`fcnn-ariann-3_threading.py`和`procedure_threading.py`无法正确识别，导致报错。具体报错信息如下：
  + `RuntimeError: The size of tensor a (10) must match the size of tensor b (0) at non-singleton dimension 1`：这是因为`p = mp.Process(target=train, args=(rank, args, private_train_loader, model, epoch))`中无法正确地将参数`private_train_loader`传递到训练函数中，为此，我们将`procedure_threading.py`中的训练函数`train`和`train_epoch`直接放到主函数中
  + 之后又报错`RuntimeError: matrices expected, got 1D, 2D tensors at /pytorch/aten/src/TH/generic/THTensorMath.cpp:131`，详细信息如下

<details>
<summary>下面是报错详细信息，请点击展开:mag:</summary>
  
```
(pysyft) root@yangpeng:~/yp_workplace/fssnn# python fcnn-ariann-3_threading.py
================================================
(AriaNN) Ciphertext Training over 15 epochs
model:           Fully Connected Neural Network
dataset:         MNIST
batch_size:      128
================================================
Process Process-1:
Traceback (most recent call last):
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 439, in handle_func_command
    cmd, args_, kwargs_, return_args_type=True
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 170, in unwrap_args_from_function
    new_args = args_hook_function(args_)
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 360, in <lambda>
    return lambda x: f(lambdas, x)
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 538, in three_fold
    lambdas[0](args_[0], **kwargs),
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 335, in <lambda>
    else lambda i: forward_func[type(i)](i)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/hook/hook_args.py", line 27, in <lambda>
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/hook/hook_args.py", line 27, in <genexpr>
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
syft.exceptions.PureFrameworkTensorFoundError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/root/miniconda3/envs/pysyft/lib/python3.7/multiprocessing/process.py", line 297, in _bootstrap
    self.run()
  File "/root/miniconda3/envs/pysyft/lib/python3.7/multiprocessing/process.py", line 99, in run
    self._target(*self._args, **self._kwargs)
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 104, in train
    train_epoch(args, model, private_train_loader, optimizer, epoch)
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 129, in train_epoch
    loss[0] = forward(optimizer, model, data, target)   # 前向传播
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 119, in forward
    output = model(data)
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 33, in forward
    x = F.relu(self.fc1(x))          # 此处的relu函数是秘密协议中的relu函数，不能自定义
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/modules/linear.py", line 87, in forward
    return F.linear(input, self.weight, self.bias)
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook.py", line 345, in overloaded_func
    response = handle_func_command(command)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 482, in handle_func_command
    response = cls._get_response(cmd, args_, kwargs_)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 516, in _get_response
    response = command_method(*args_, **kwargs_)
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/functional.py", line 1370, in linear
    ret = torch.addmm(bias, input, weight.t())
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook.py", line 345, in overloaded_func
    response = handle_func_command(command)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 482, in handle_func_command
    response = cls._get_response(cmd, args_, kwargs_)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 516, in _get_response
    response = command_method(*args_, **kwargs_)
RuntimeError: matrices expected, got 1D, 2D tensors at /pytorch/aten/src/TH/generic/THTensorMath.cpp:131
Process Process-3:
Traceback (most recent call last):
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 439, in handle_func_command
    cmd, args_, kwargs_, return_args_type=True
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 170, in unwrap_args_from_function
    new_args = args_hook_function(args_)
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 360, in <lambda>
    return lambda x: f(lambdas, x)
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 538, in three_fold
    lambdas[0](args_[0], **kwargs),
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 335, in <lambda>
    else lambda i: forward_func[type(i)](i)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/hook/hook_args.py", line 27, in <lambda>
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/hook/hook_args.py", line 27, in <genexpr>
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
syft.exceptions.PureFrameworkTensorFoundError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/root/miniconda3/envs/pysyft/lib/python3.7/multiprocessing/process.py", line 297, in _bootstrap
    self.run()
  File "/root/miniconda3/envs/pysyft/lib/python3.7/multiprocessing/process.py", line 99, in run
    self._target(*self._args, **self._kwargs)
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 104, in train
    train_epoch(args, model, private_train_loader, optimizer, epoch)
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 129, in train_epoch
    loss[0] = forward(optimizer, model, data, target)   # 前向传播
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 119, in forward
    output = model(data)
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 33, in forward
    x = F.relu(self.fc1(x))          # 此处的relu函数是秘密协议中的relu函数，不能自定义
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/modules/linear.py", line 87, in forward
    return F.linear(input, self.weight, self.bias)
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook.py", line 345, in overloaded_func
    response = handle_func_command(command)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 482, in handle_func_command
    response = cls._get_response(cmd, args_, kwargs_)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 516, in _get_response
    response = command_method(*args_, **kwargs_)
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/functional.py", line 1370, in linear
    ret = torch.addmm(bias, input, weight.t())
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook.py", line 345, in overloaded_func
    response = handle_func_command(command)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 482, in handle_func_command
    response = cls._get_response(cmd, args_, kwargs_)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 516, in _get_response
    response = command_method(*args_, **kwargs_)
RuntimeError: matrices expected, got 1D, 2D tensors at /pytorch/aten/src/TH/generic/THTensorMath.cpp:131
Process Process-2:
Traceback (most recent call last):
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 439, in handle_func_command
    cmd, args_, kwargs_, return_args_type=True
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 170, in unwrap_args_from_function
    new_args = args_hook_function(args_)
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 360, in <lambda>
    return lambda x: f(lambdas, x)
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 538, in three_fold
    lambdas[0](args_[0], **kwargs),
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook_args.py", line 335, in <lambda>
    else lambda i: forward_func[type(i)](i)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/hook/hook_args.py", line 27, in <lambda>
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/hook/hook_args.py", line 27, in <genexpr>
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
syft.exceptions.PureFrameworkTensorFoundError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/root/miniconda3/envs/pysyft/lib/python3.7/multiprocessing/process.py", line 297, in _bootstrap
    self.run()
  File "/root/miniconda3/envs/pysyft/lib/python3.7/multiprocessing/process.py", line 99, in run
    self._target(*self._args, **self._kwargs)
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 104, in train
    train_epoch(args, model, private_train_loader, optimizer, epoch)
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 129, in train_epoch
    loss[0] = forward(optimizer, model, data, target)   # 前向传播
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 119, in forward
    output = model(data)
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/root/yp_workplace/fssnn/fcnn-ariann-3_threading.py", line 33, in forward
    x = F.relu(self.fc1(x))          # 此处的relu函数是秘密协议中的relu函数，不能自定义
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/modules/linear.py", line 87, in forward
    return F.linear(input, self.weight, self.bias)
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook.py", line 345, in overloaded_func
    response = handle_func_command(command)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 482, in handle_func_command
    response = cls._get_response(cmd, args_, kwargs_)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 516, in _get_response
    response = command_method(*args_, **kwargs_)
  File "/root/miniconda3/envs/pysyft/lib/python3.7/site-packages/torch/nn/functional.py", line 1370, in linear
    ret = torch.addmm(bias, input, weight.t())
  File "/root/yp_workplace/ariann/PySyft/syft/generic/frameworks/hook/hook.py", line 345, in overloaded_func
    response = handle_func_command(command)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 482, in handle_func_command
    response = cls._get_response(cmd, args_, kwargs_)
  File "/root/yp_workplace/ariann/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 516, in _get_response
    response = command_method(*args_, **kwargs_)
RuntimeError: matrices expected, got 1D, 2D tensors at /pytorch/aten/src/TH/generic/THTensorMath.cpp:131
  
```
</details>

#### 2.2 我们的目标

我们希望减少一个epoch的训练时间，以便和参照论文[AriaNN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing](https://petsymposium.org/popets/2022/popets-2022-0015.php)的数据是一个数量级，这是我们最终的目标。
  
