from os import XATTR_SIZE_MAX
import torch
import syft as sy
hook = sy.TorchHook(torch)

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

# x = torch.tensor([2, 3, 4, 1]).share(bob,alice, crypto_provider=crypto_provider)
# x_max = x.max()
# print(x_max.get())

# x = torch.tensor([[2, 3], [4, 1]]).share(bob,alice, crypto_provider=crypto_provider)
# max_values = x.max(dim=0)
# print(max_values.get())

x = torch.tensor([2,3,4,1]).share(bob,alice,crypto_provider=crypto_provider, protocol="fss")
print(x)