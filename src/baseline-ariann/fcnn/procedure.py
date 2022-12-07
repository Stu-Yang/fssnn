import time

import torch
import syft as sy


# 自定义均方误差函数
def mse_loss_mean(output, target):
    batch_size = output.shape[0]
    loss = ((output - target) ** 2).sum().refresh() / batch_size    # Refresh shares by adding shares of zero
    return loss

def mse_loss_sum(output, target):
    loss = ((output - target) ** 2).sum()
    return loss

# 定义训练过程
def train(args, model, private_train_loader, optimizer, epoch):   # 训练函数
    model.train()
    times = []

    try:
        n_items = (len(private_train_loader) - 1) * args.batch_size + len(
            private_train_loader[-1][1]
        )
    except TypeError:
        n_items = len(private_train_loader.dataset)

    for batch_idx, (data, target) in enumerate(private_train_loader):    # 小批量训�?
        start_time = time.time()

        def forward(optimizer, model, data, target):    # 前向传播，其定义了每次执行的计算步骤，最后返回loss函数
            optimizer.zero_grad()    # 清空所有被优化过的Variable的梯度

            output = model(data)     # 这里的model()会调用root/gsq_workplace/arinn/ariann/models.py中的forward函数
            
            loss_enc = mse_loss_mean(output, target)    # 计算误差
            
            return loss_enc

        loss = [10e10]                     # 初始化loss函数
        loss_dec = torch.tensor([10e10])

        while loss_dec.abs() > 10:
            loss[0] = forward(optimizer, model, data, target)   # 这里调用的forward函数是line 47定义的forward函数

            loss_dec = loss[0].copy()
            if loss_dec.is_wrapper:
                loss_dec = loss_dec.get().float_precision()

            if loss_dec.abs() > 10:
                print(f'⚠️ #{batch_idx} loss:{loss_dec.item()} RETRY...')

        loss[0].backward()    # 反向传播

        optimizer.step()      # 进行单次优化 (参数更新).
        tot_time = time.time() - start_time
        times.append(tot_time)

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s ({:.3f}s/item) [{:.3f}]".format(
                    epoch,
                    batch_idx * args.batch_size,
                    n_items,
                    100.0 * batch_idx / len(private_train_loader),
                    loss_dec.item(),
                    tot_time,
                    tot_time / args.batch_size,
                    args.batch_size,
                )
            )

    print()
    return torch.tensor(times).mean().item()

# 模型测试过程
def test(args, model, private_test_loader):    # 测试函数
    model.eval()
    correct = 0
    times = 0
    real_times = 0  # with the argmax
    i = 0
    try:
        n_items = (len(private_test_loader) - 1) * args.test_batch_size + len(
            private_test_loader[-1][1]
        )
    except TypeError:
        n_items = len(private_test_loader.dataset)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(private_test_loader):
            i += 1
            start_time = time.time()

            output = model(data)

            times += time.time() - start_time
            pred = output.argmax(dim=1)
            real_times += time.time() - start_time
            te=pred.eq(target.view_as(pred))
            p1=pred.copy().get()
            p2=target.copy().get()
            ta=p1.eq(p2)
            #print("pred\n real\n diff{}".format(ta.sum()))
            correct += ta.sum()
            if batch_idx % (args.log_interval * 4) == 0 and correct.is_wrapper:
                c = correct.copy().float_precision()
                ni = i * args.test_batch_size
                print(
                    "Accuracy: {}/{} ({:.0f}%) \tTime / item: {:.4f}s".format(
                        int(c.item()),
                        ni,
                        100.0 * c.item() / ni,
                        times / ni,
                    )
                )
    if correct.is_wrapper:
        correct = correct.float_precision()

    print(
        "TEST Accuracy: {}/{} ({:.2f}%) \tTime /item: {:.4f}s \tTime w. argmax /item: {:.4f}s [{:.3f}]\n".format(
            correct.item(),
            n_items,
            100.0 * correct.item() / n_items,
            times / n_items,
            real_times / n_items,
            args.test_batch_size,
        )
    )

    return times, round(100.0 * correct.item() / n_items, 1)