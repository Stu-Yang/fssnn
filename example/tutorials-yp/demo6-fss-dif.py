from syft.frameworks.torch.mpc.fss import DIF, N
import numpy as np

keys_a, keys_b = DIF.keygen(n_values=4)

alpha_a = np.frombuffer(np.ascontiguousarray(keys_a[:, 0:N]), dtype=np.uint32).astype(np.uint64)
alpha_b = np.frombuffer(np.ascontiguousarray(keys_b[:, 0:N]), dtype=np.uint32).astype(np.uint64)

x = np.array([[1, 1], [-1, 1]])
print("x = \n", x)

x_masked = (x + alpha_a.reshape(x.shape) + alpha_b.reshape(x.shape)).astype(np.uint64)

y0 = DIF.eval(0, x_masked, keys_a)
y1 = DIF.eval(1, x_masked, keys_b)

print("\ny0 + y1 = \n", y0+y1)

