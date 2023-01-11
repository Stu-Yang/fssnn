import numpy as np

def mat_mul(A, B):
    assert A.shape[1] == B.shape[0]
    return np.dot(A, B)
    

if __name__ == "__main__":
    A = np.arange(9).reshape(3, 3)
    B = np.arange(12).reshape(3, 4)
    C = mat_mul(A, B)
    print("C = A * B = ", C)





# import torch
# import syft as sy  # import the Pysyft library
# hook = sy.TorchHook(torch)  # hook PyTorch to add extra functionalities like Federated and Encrypted Learning

# # # simulation functions
# # def connect_to_workers(n_workers):
# #     return [
# #         sy.VirtualWorker(hook, id=f"worker{i+1}")
# #         for i in range(n_workers)
# #     ]
# # def connect_to_crypto_provider():
# #     return sy.VirtualWorker(hook, id="crypto_provider")

# # workers = connect_to_workers(n_workers=2)
# # crypto_provider = connect_to_crypto_provider()

# # print("*workers = ", *workers)
# # print("crypto_provider = ", crypto_provider)



# bob = sy.VirtualWorker(hook, id="bob")
# alice = sy.VirtualWorker(hook, id="alice")
# crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
# workers = [alice, bob]
# print("*workers = ", *workers)
# print("crypto_provider = ", crypto_provider)