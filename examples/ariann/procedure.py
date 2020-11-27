import cProfile
import io
import os
import pstats
import time

import torch


def profile(func):
    """A gentle profiler"""

    def wrapper(args_, *args, **kwargs):
        if args_.verbose:
            pr = cProfile.Profile()
            pr.enable()
            retval = func(args_, *args, **kwargs)
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
            ps.print_stats(0.1)
            print(s.getvalue())
            return retval
        else:
            return func(args_, *args, **kwargs)

    return wrapper


def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    times = []

    try:
        n_items = (len(private_train_loader) - 1) * args.batch_size + len(
            private_train_loader[-1][1]
        )
    except TypeError:
        n_items = len(private_train_loader.dataset)

    for batch_idx, (data, target) in enumerate(private_train_loader):
        start_time = time.time()

        optimizer.zero_grad()

        output = model(data)

        # loss = F.nll_loss(output, target)  <-- not possible here
        batch_size = output.shape[0]

        loss = ((output - target) ** 2).sum() / batch_size

        loss.backward()

        optimizer.step()
        tot_time = time.time() - start_time
        times.append(tot_time)

        if batch_idx % args.log_interval == 0:
            if loss.is_wrapper:
                if not args.fp_only:
                    loss = loss.get()
                loss = loss.float_precision()
            if args.train:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s ({:.3f}s/item) [{:.3f}]".format(
                        epoch,
                        batch_idx * args.batch_size,
                        n_items,
                        100.0 * batch_idx / len(private_train_loader),
                        loss.item(),
                        tot_time,
                        tot_time / args.batch_size,
                        args.batch_size,
                    )
                )

    print()
    return torch.tensor(times).mean().item()


@profile
def test(args, model, private_test_loader):
    model.eval()
    correct = 0
    times = 0
    real_times = 0  # with the argmax
    i = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(private_test_loader):
            i += 1
            start_time = time.time()
            output = model(data)
            times += time.time() - start_time
            pred = output.argmax(dim=1)
            real_times += time.time() - start_time
            correct += pred.eq(target.view_as(pred)).sum()
            if batch_idx % args.log_interval == 0 and correct.is_wrapper:
                if args.fp_only:
                    c = correct.copy().float_precision()
                else:
                    c = correct.copy().get().float_precision()
                ni = i * args.test_batch_size
                if args.test:
                    print(
                        "Accuracy: {}/{} ({:.0f}%) \tTime / item: {:.4f}s".format(
                            int(c.item()),
                            ni,
                            100.0 * c.item() / ni,
                            times / ni,
                        )
                    )

    if correct.is_wrapper:
        if args.fp_only:
            correct = correct.float_precision()
        else:
            correct = correct.get().float_precision()

    try:
        n_items = (len(private_test_loader) - 1) * args.test_batch_size + len(
            private_test_loader[-1][1]
        )
    except TypeError:
        n_items = len(private_test_loader.dataset)

    if args.test:
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
