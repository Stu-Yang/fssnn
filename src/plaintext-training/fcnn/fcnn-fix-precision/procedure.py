import time
import torch
import syft as sy

# 本文件主要参考自https://github.com/LaRiffle/ariann/blob/main/procedure.py

# 定义one-hot编码函数
def one_hot_of(index_tensor):
    """
    Transform to one hot tensor

    Example:
        [0, 3, 9]
        =>
        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]

    """
    onehot_tensor = torch.zeros(*index_tensor.shape, 10)  # 10 classes for MNIST
    onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
    return onehot_tensor


# 自定义均方误差函数
def mse_loss_mean(output, target):
    batch_size = output.shape[0]
    loss = ((output - target) ** 2).sum() / batch_size
    return loss

def mse_loss_sum(output, target):
    loss = ((output - target) ** 2).sum()
    return loss


# 定义训练过程，参考https://github.com/LaRiffle/ariann/blob/main/procedure.py
def train(args, model, train_loader, optimizer, epoch):
    model.train()
    times = []

    try:
        n_items = (len(train_loader) - 1) * args.batch_size + len(
            train_loader[-1][1]
        )
    except TypeError:
        n_items = len(train_loader.dataset)

    for batch_idx, (data, target) in enumerate(train_loader):
        start_time = time.time()

        def forward(optimizer, model, data, target):
            optimizer.zero_grad()

            output = model(data)

            loss_enc = mse_loss_mean(output, target)

            return loss_enc

        loss = [10e10]
        loss_dec = torch.tensor([10e10])

        while loss_dec.abs() > 10:
            loss[0] = forward(optimizer, model, data, target)

            loss_dec = loss[0].copy()
            if loss_dec.is_wrapper:
                loss_dec = loss_dec.float_precision()

            if loss_dec.abs() > 10:
                print(f'⚠️ #{batch_idx} loss:{loss_dec.item()} RETRY...')

        loss[0].backward()

        optimizer.step()
        tot_time = time.time() - start_time
        times.append(tot_time)

        if batch_idx % args.log_interval == 0:
            print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s ({:.3f}s/item) [{:.3f}]".format(
                        epoch,
                        batch_idx * args.batch_size,
                        n_items,
                        100.0 * batch_idx / len(train_loader),
                        loss_dec.item(),
                        tot_time,
                        tot_time / args.batch_size,
                        args.batch_size,
                    )
                )
    print()
    return torch.tensor(times).mean().item()


# 定义模型测试过程，参考https://github.com/LaRiffle/ariann/blob/main/procedure.py
def test(args, model, test_loader):
    model.eval()
    correct = 0
    times = 0
    real_times = 0  # with the argmax
    i = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            i += 1
            start_time = time.time()

            output = model(data)

            times += time.time() - start_time
            pred = output.argmax(dim=1)
            real_times += time.time() - start_time
            correct += pred.eq(target.view_as(pred)).sum()
            if batch_idx % args.log_interval == 0 and correct.is_wrapper:
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

    try:
        n_items = (len(test_loader) - 1) * args.test_batch_size + len(
            test_loader[-1][1]
        )
    except TypeError:
        n_items = len(test_loader.dataset)

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
