import torch as th
import torch
from typing import Tuple
import math

import syft as sy
from .cuda.tensor import CUDALongTensor

if torch.cuda.is_available():
    import torchcsprng as csprng

    generator = csprng.create_random_device_generator("/dev/urandom")

dtypes = {
    "torch.float32": torch.float32,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.bool": torch.bool,
}


def build_triple(
    op: str,
    kwargs_: dict,
    shape: Tuple[th.Size, th.Size],
    n_workers: int,
    torch_dtype: th.dtype,
    field: int,
):
    """
    Generates and shares a multiplication triple (a, b, c)

    Args:
        op (str): 'mul' or 'matmul': the op ° which ensures a ° b = c
        shape (Tuple[th.Size, th.Size]): the shapes of a and b
        n_workers (int): number of workers
        torch_dtype (th.dtype): the type of the shares
        field (int): the field for the randomness

    Returns:
        a triple of shares (a_sh, b_sh, c_sh) per worker where a_sh is a share of a
    """

    left_shape, right_shape = shape
    cmd = getattr(th, op)
    low_bound, high_bound = -(field // 2), (field - 1) // 2
    a_b_loc=-1
    if op == "matmul":
        matmul_a_b = getattr(sy.local_worker.crypto_store, "matmul_a_b")
        layers=getattr(sy.local_worker.crypto_store,"layers")
        loc=getattr(sy.local_worker.crypto_store,"loc")
        if loc>layers:
            f_layers=layers - math.ceil((loc - layers) / 2)   #layers-1,...1,0
            a_b_loc=(loc%2)^(layers%2)#0 重用a 1  重用b  #相关矩阵

    if isinstance(torch_dtype, str):
        torch_dtype = dtypes[torch_dtype]
    if torch.cuda.is_available():
        if a_b_loc==0:
            a=matmul_a_b[f_layers][0]
        else:
            a = th.empty(*left_shape, dtype=torch_dtype, device="cuda").random_(
                low_bound, high_bound, generator=generator
            )
        if a_b_loc==1:
            b=matmul_a_b[f_layers][1]
        else:
            b = th.empty(*right_shape, dtype=torch_dtype, device="cuda").random_(
                low_bound, high_bound, generator=generator
            )
    else:
        if a_b_loc==0:
            a=matmul_a_b[f_layers][0].t()
        else:
            a = th.randint(low_bound, high_bound, left_shape, dtype=torch_dtype)
        if a_b_loc==1:
            b=matmul_a_b[f_layers][1].t()
        else:
            b = th.randint(low_bound, high_bound, right_shape, dtype=torch_dtype)
    if a_b_loc<0 and op =="matmul":
        matmul_a_b[len(matmul_a_b)-1].extend([a,b])
        matmul_a_b.append([])
    if op == "mul":
        if b.numel() == a.numel():
            # examples:
            #   torch.tensor([3]) * torch.tensor(3) = tensor([9])
            #   torch.tensor([3]) * torch.tensor([[3]]) = tensor([[9]])
            if len(a.shape) == len(b.shape):
                c = cmd(a, b)
            elif len(a.shape) > len(b.shape):
                b_ = b.reshape_as(a)
                c = cmd(a, b_)
            else:  # len(a.shape) < len(b.shape):
                a_ = a.reshape_as(b)
                c = cmd(a_, b)
        else:
            c = cmd(a, b)
    elif op in {"matmul", "conv2d", "conv_transpose2d"}:
        if th.cuda.is_available():
            cmd = getattr(CUDALongTensor, op)
            a_ = CUDALongTensor(a)
            b_ = CUDALongTensor(b)
            c_ = cmd(a_, b_, **kwargs_)
            c = c_._tensor
        else:
            cmd = getattr(th, op)
            c = cmd(a, b, **kwargs_)
    else:
        raise ValueError

    helper = sy.AdditiveSharingTensor(field=field)

    shares_worker = [[0, 0, 0] for _ in range(n_workers)]
    for i, tensor in enumerate([a, b, c]):
        shares = helper.generate_shares(secret=tensor, n_workers=n_workers, random_type=torch_dtype)
        for w_id in range(n_workers):
            if a_b_loc==i:
                shares_worker[w_id][i] = 0
                # shares_worker[w_id][i] = shares[w_id]
            else:
                shares_worker[w_id][i] = shares[w_id]

    return shares_worker
