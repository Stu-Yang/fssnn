import torch
import syft as sy
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")

x = torch.tensor([1,2,3,4,5])
y = torch.tensor([1,1,1,1,1])

x_ptr = x.send(bob)     # 将数据x发送给bob
y_ptr = y.send(bob)     # 将数据y发送给bob
z_ptr = x_ptr + y_ptr   # bob计算z=x+y
z = z_ptr.get()         # 本地获取z，此时bob不再拥有z

print(x_ptr)
print(x_ptr.location)   #
print(bob)


# u_ptr = z_ptr + y_ptr   # 错误！bob只有y，没有z
# u = u_ptr.get()
# print(u)


# z_ptr = z.send(bob)     # 将z重新发送给bob
# u_ptr = z_ptr + y_ptr
# u = u_ptr.get()
# print(u)