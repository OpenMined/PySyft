import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import logging

import syft as sy  # <-- NEW: import the Pysyft library
from syft.workers import WebsocketClientWorker
from syft.workers import VirtualWorker
from syft.frameworks.torch.federated import utils


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_on_batches(worker, batches, model_in, device, lr):
    # model = copy.deepcopy(model_in)
    # model = model_in
    # model = type(model_in)()  # get a new instance
    # model.load_state_dict(model_in.state_dict())  # copy weights and stuff
    # model.to(device)
    model = model_in.copy()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # TODO momentum is not supported at the moment

    model.train()
    model.send(worker)  # <-- NEW: send the model to the right location
    log_interval = 10
    loss_local = False

    for batch_idx, (data, target) in enumerate(batches):
        loss_local = False
        # data, target = batches[batch_idx]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            loss_local = True
            print(
                "Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    worker.id,
                    batch_idx,
                    len(batches),
                    100.0 * batch_idx / len(batches),
                    loss.item(),
                )
            )
    if not loss_local:
        loss = loss.get()  # <-- NEW: get the loss back
    model.get()  # <-- NEW: get the model back
    return model, loss


def train(model, device, federated_train_loader, lr):
    print("Starting train()")
    model.train()
    batches = utils.extract_batches_per_worker(federated_train_loader)
    print("After extract batches()")
    limit_to_one_worker = False
    keys = list(batches.keys())

    if limit_to_one_worker:
        # artificially limit to one worker
        keys = list(batches.keys())
        batches = {keys[0]: batches[keys[0]]}
    nr_batches = 20

    models = {}
    loss_values = {}

    print("calculating available batches")
    available_batches = max(map(lambda x: len(x[1]), batches.items()))
    for start_idx in range(0, available_batches, nr_batches):
        print("Starting training round, batches [{}, {}]".format(start_idx, start_idx + nr_batches))

        for worker in batches:
            curr_batches = batches[worker][start_idx : start_idx + nr_batches]
            if curr_batches:
                models[worker], loss_values[worker] = train_on_batches(
                    worker,
                    curr_batches,
                    # model.copy(), device, optimizer)
                    model,
                    device,
                    lr,
                )
        model = utils.federated_avg(models)
        # model = models[keys[0]]
    return model


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def main():
    class Arguments:
        def __init__(self):
            self.batch_size = 64
            self.test_batch_size = 1000
            self.epochs = 10
            self.lr = 0.01
            self.momentum = 0.5
            self.no_cuda = False
            self.seed = 1
            self.log_interval = 30
            self.save_model = False
            self.verbose = False
            self.use_virtual = False

    args = Arguments()

    hook = sy.TorchHook(
        torch
    )  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning

    if args.use_virtual:
        alice = VirtualWorker(id="alice", hook=hook, verbose=args.verbose)
        bob = VirtualWorker(id="bob", hook=hook, verbose=args.verbose)
        charlie = VirtualWorker(id="charlie", hook=hook, verbose=args.verbose)
    else:
        kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
        alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
        bob = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
        charlie = WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)

    workers = [alice, bob, charlie]

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    federated_train_loader = sy.FederatedDataLoader(  # <-- this is now a FederatedDataLoader
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ).federate(tuple(workers)),
        # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )

    model = Net().to(device)

    for epoch in range(1, args.epochs + 1):
        print("Starting epoch {}/{}".format(epoch, args.epochs))
        model = train(model, device, federated_train_loader, args.lr)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d, p:%(process)d) - %(message)s"
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.DEBUG)
    websockets_logger.addHandler(logging.StreamHandler())

    main()
