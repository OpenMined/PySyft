import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

MODEL_PATH = "/content/PySyft/data/models/"


def save(model, name):
    torch.save(model.state_dict(), MODEL_PATH + name)


def load(model, name):
    model.load_state_dict(torch.load(MODEL_PATH + name))
    model.eval()
    return model


def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    times = []

    n_items = (len(private_train_loader) - 1) * args.batch_size + len(private_train_loader[-1][1])

    for batch_idx, (data, target) in enumerate(
        private_train_loader
    ):  # <-- now it is a private dataset
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
                loss = loss.get().float_precision()
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

    return torch.tensor(times).mean().item()


def test(args, model, private_test_loader):
    model.eval()
    correct = 0
    times = 0
    real_times = 0  # with the argmax
    i = 0
    with torch.no_grad():
        for data, target in private_test_loader:
            i += 1
            start_time = time.time()
            output = model(data)
            times += time.time() - start_time
            pred = output.argmax(dim=1)
            real_times += time.time() - start_time
            correct += pred.eq(target.view_as(pred)).sum()
            if correct.is_wrapper:
                c = correct.copy().get().float_precision()
                ni = i * args.test_batch_size
                print(c, ni, round(100 * c.item() / ni, 2), "%")

    if correct.is_wrapper:
        correct = correct.get().float_precision()

    n_items = (len(private_test_loader) - 1) * args.test_batch_size + len(
        private_test_loader[-1][1]
    )
    print(
        "\nTest set: Accuracy: {}/{} ({:.0f}%) \tTime /item: {:.4f}s \tTo time /item: {:.4f}s [{:.3f}]\n".format(
            correct.item(),
            n_items,
            100.0 * correct.item() / n_items,
            times / n_items,
            real_times / n_items,
            args.test_batch_size,
        )
    )

    return torch.tensor(times).mean().item(), round(100.0 * correct.item() / n_items, 1)


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


def get_private_data_loaders(workers, args, kwargs):
    def secret_share(tensor):
        """
        Transform to fixed precision and secret share a tensor
        """
        return tensor.fix_precision(precision_fractional=args.precision_fractional).share(
            *workers, **kwargs
        )

    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, download=True, transform=transformation),
        batch_size=args.batch_size,
    )

    private_train_loader = []
    for i, (data, target) in enumerate(train_loader):
        if args.n_train_items >= 0 and i >= args.n_train_items / args.batch_size:
            break
        private_train_loader.append((secret_share(data), secret_share(one_hot_of(target))))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, download=True, transform=transformation),
        batch_size=args.test_batch_size,
    )

    private_test_loader = []
    for i, (data, target) in enumerate(test_loader):
        if args.n_test_items >= 0 and i >= args.n_test_items / args.test_batch_size:
            break
        private_test_loader.append((secret_share(data), secret_share(target.float())))

    return private_train_loader, private_test_loader


def get_public_data_loaders(workers, args, kwargs):

    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, download=True, transform=transformation),
        batch_size=args.batch_size,
    )

    public_train_loader = []
    for i, (data, target) in enumerate(train_loader):
        if args.n_train_items >= 0 and i >= args.n_train_items / args.batch_size:
            break
        public_train_loader.append((data, one_hot_of(target)))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, download=True, transform=transformation),
        batch_size=args.test_batch_size,
    )

    public_test_loader = []
    for i, (data, target) in enumerate(test_loader):
        if args.n_test_items >= 0 and i >= args.n_test_items / args.test_batch_size:
            break
        public_test_loader.append((data, target))

    return public_train_loader, public_test_loader


def estimate_time(
    time_one_batch_train,
    time_one_batch_test,
    batch_size,
    epochs,
    dataset_size_train,
    dataset_size_test,
):
    train_time = dataset_size_train / batch_size * time_one_batch_train * epochs
    print("Train time\t{:.3f}s\t{:.3f}h".format(train_time, train_time / 3600))
    test_time = dataset_size_test / batch_size * time_one_batch_test
    print("Test time\t{:.3f}s\t{:.3f}h".format(test_time, test_time / 3600))
