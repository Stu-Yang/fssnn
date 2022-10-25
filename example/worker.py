from pyexpat.errors import XML_ERROR_PARTIAL_CHAR
import torch
import syft as sy
hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(hook, id="alice")      # 创建远程工作节点alice
bob = sy.VirtualWorker(hook, id="bob")          # 创建远程工作节点bob

x = torch.tensor([1,2,3,4,5])       # 本地张量数据
y = torch.tensor([1,1,1,1,1])       # 本地张量数据
z = x + y       # 在本地计算x+y，得到本地张量z
print("local z = ", z)

x_ptr = x.send(alice)   # 将x发送给alice
y_ptr = y.send(bob)     # 将y发送给bob

print("alice has ", alice._objects)
print("alice's location", x_ptr.location)


