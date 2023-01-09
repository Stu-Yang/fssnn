# 当修改matmul之后出现的错误

```shell
(fssnn) root@yp:~/yp_workplace/fssnn/src/optimized-fssnn/ariann-main# python main.py --model network1 --dataset mnist --train --epochs 15 --lr 0.01
Training over 15 epochs
model:           network1
dataset:         mnist
batch_size:      128
Traceback (most recent call last):
  File "/root/FssNN/PySyft/syft/frameworks/torch/mpc/fss.py", line 110, in fss_op
    share = remote(mask_builder, location=location)(*workers_args[i], return_value=True)
  File "/root/FssNN/PySyft/syft/generic/utils.py", line 50, in remote_exec
    location, *command, return_ids=response_ids, return_value=return_value
  File "/root/FssNN/PySyft/syft/workers/base.py", line 637, in send_command
    ret_val = self.send_msg(message, location=recipient)
  File "/root/FssNN/PySyft/syft/workers/base.py", line 335, in send_msg
    bin_response = self._send_msg(bin_message, location)
  File "/root/FssNN/PySyft/syft/workers/virtual.py", line 12, in _send_msg
    return location._recv_msg(message)
  File "/root/FssNN/PySyft/syft/workers/virtual.py", line 22, in _recv_msg
    return self.recv_msg(message)
  File "/root/FssNN/PySyft/syft/workers/base.py", line 416, in recv_msg
    response = handler.handle(msg)
  File "/root/FssNN/PySyft/syft/generic/abstract/message_handler.py", line 20, in handle
    return self.routing_table[type(msg)](msg)
  File "/root/FssNN/PySyft/syft/workers/message_handler.py", line 55, in execute_tensor_command
    return self.execute_computation_action(cmd.action)
  File "/root/FssNN/PySyft/syft/workers/message_handler.py", line 119, in execute_computation_action
    response = command(*args_, **kwargs_)
  File "/root/FssNN/PySyft/syft/frameworks/torch/mpc/fss.py", line 165, in mask_builder
    keys = worker.crypto_store.get_keys(f"fss_{op}", n_instances=numel, remove=False)
  File "/root/FssNN/PySyft/syft/frameworks/torch/mpc/primitives.py", line 147, in get_keys
    self, available_instances, n_instances=n_instances, op=op, **kwargs
syft.exceptions.EmptyCryptoPrimitiveStoreError: You tried to run a crypto protocol on worker alice but its crypto_store doesn't have enough primitives left for the type 'fss_comp None' (16384 were requested while only -1 are available). Use your crypto_provider to `provide_primitives` to your worker.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "main.py", line 326, in <module>
    run(args)
  File "main.py", line 95, in run
    (trianing_times_epoch, trianing_comm_epoch) = train(args, model, private_train_loader, optimizer, epoch)
  File "/root/yp_workplace/fssnn/src/optimized-fssnn/ariann-main/procedure.py", line 64, in train
    loss[0] = forward(optimizer, model, data, target)
  File "/root/yp_workplace/fssnn/src/optimized-fssnn/ariann-main/procedure.py", line 50, in forward
    output = model(data)
  File "/root/gsq_workplace/anconda3/envs/fssnn/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/root/yp_workplace/fssnn/src/optimized-fssnn/ariann-main/models.py", line 20, in forward
    x = F.relu(x)
  File "/root/FssNN/PySyft/syft/generic/frameworks/hook/hook.py", line 346, in overloaded_func
    response = handle_func_command(command)
  File "/root/FssNN/PySyft/syft/frameworks/torch/tensors/interpreters/native.py", line 456, in handle_func_command
    response = new_type.handle_func_command(new_command)
  File "/root/FssNN/PySyft/syft/frameworks/torch/tensors/interpreters/autograd.py", line 333, in handle_func_command
    return cmd(*args_, **kwargs_)
  File "/root/FssNN/PySyft/syft/frameworks/torch/tensors/interpreters/autograd.py", line 278, in relu
    return tensor.relu()
  File "/root/FssNN/PySyft/syft/frameworks/torch/tensors/interpreters/autograd.py", line 181, in method_with_grad
    result = getattr(new_self, name)(*new_args, **new_kwargs)
  File "/root/FssNN/PySyft/syft/generic/frameworks/hook/hook.py", line 112, in overloaded_syft_method
    response = getattr(new_self, attr)(*new_args, **new_kwargs)
  File "/root/FssNN/PySyft/syft/frameworks/torch/mpc/__init__.py", line 35, in method
    return f(self, *args, **kwargs)
  File "/root/FssNN/PySyft/syft/frameworks/torch/tensors/interpreters/additive_shared.py", line 975, in relu
    return self * (self >= 0)
  File "/root/FssNN/PySyft/syft/frameworks/torch/mpc/__init__.py", line 35, in method
    return f(self, *args, **kwargs)
  File "/root/FssNN/PySyft/syft/frameworks/torch/tensors/interpreters/additive_shared.py", line 1005, in __ge__
    return fss.le(other, self)
  File "/root/FssNN/PySyft/syft/frameworks/torch/mpc/fss.py", line 222, in le
    return fss_op(x1, x2, "comp")
  File "/root/FssNN/PySyft/syft/frameworks/torch/mpc/fss.py", line 115, in fss_op
    sy.local_worker.crypto_store.provide_primitives(workers=locations, kwargs_={}, **e.kwargs_)
  File "/root/FssNN/PySyft/syft/frameworks/torch/mpc/primitives.py", line 184, in provide_primitives
    kwargs_=kwargs_, n_party=len(workers), n_instances=n_instances, **kwargs
TypeError: build_separate_fss_keys() got an unexpected keyword argument 'loc'
```

# 如何去复现？

进入到服务器，并在终端依次输入：
```shell
root@yp:~# conda activate fssnn
(fssnn) root@yp:~# cd /root/yp_workplace/fssnn/src/optimized-fssnn/ariann-main
(fssnn) root@yp:~/yp_workplace/fssnn/src/optimized-fssnn/ariann-main# python main.py --model network1 --dataset mnist --train --epochs 15 --lr 0.01
Training over 15 epochs
model:           network1
dataset:         mnist
batch_size:      128
Traceback (most recent call last):
  File "/root/FssNN/PySyft/syft/frameworks/torch/mpc/fss.py", line 110, in fss_op
  ...

```