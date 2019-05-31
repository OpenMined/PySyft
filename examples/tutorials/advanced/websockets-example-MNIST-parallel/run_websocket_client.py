import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import logging
import argparse
import sys
import asyncio

FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
LOG_LEVEL = logging.INFO
logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

import syft as sy

from syft import workers
from syft.frameworks.torch import pointers
from syft.frameworks.torch.federated import utils

logger = logging.getLogger(__name__)

LOG_INTERVAL = 25


# Loss function
@torch.jit.script
def loss_fn(output, target):
    return F.nll_loss(output, target)


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


def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=32, help="batch size used for the test data"
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train")
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=50,
        help="number of training steps performed on each remote worker before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will be started in verbose mode",
    )
    parser.add_argument(
        "--use_virtual", action="store_true", help="if set, virtual workers will be used"
    )

    args = parser.parse_args(args=args)
    return args


def accuracy(pred_softmax, target):
    nr_elems = len(target)
    pred = pred_softmax.argmax(dim=1)
    logger.debug("predicted: %s", pred)
    logger.debug("target:    %s", target)
    return (pred == target).sum().numpy() / float(nr_elems)


async def fit_model_on_worker(worker, traced_model, batch_size, curr_epoch, max_nr_batches):
    # Create and send train config
    train_config = sy.TrainConfig(
        batch_size=batch_size, shuffle=True, max_nr_batches=max_nr_batches, epochs=1
    )
    train_config.send(worker, traced_model=traced_model, traced_loss_fn=loss_fn)
    logger.info("Training round %s, calling fit on worker: %s", curr_epoch, worker.id)
    loss = await worker.fit(dataset="mnist", return_ids=[0])
    logger.info("Training round: %s, worker: %s, avg_loss: %s", curr_epoch, worker.id, loss.mean())
    # logger.debug("Worker state: %s", worker)
    # logger.debug("Worker objects: \n%s", worker.list_objects_remote())
    # logger.debug("loss: mean %s, max %s, min %s", loss.mean(), loss.max(), loss.min())
    model = None
    if not torch.isnan(loss).any():
        model = train_config.model_ptr.get().obj
    return worker.id, model, loss


async def main():
    args = define_and_get_arguments()
    # args.use_virtual = True

    hook = sy.TorchHook(torch)
    me = hook.local_worker

    if args.use_virtual:
        alice = sy.workers.VirtualWorker(id="alice", hook=hook, verbose=args.verbose)
        bob = sy.workers.VirtualWorker(id="bob", hook=hook, verbose=args.verbose)
        charlie = sy.workers.VirtualWorker(id="charlie", hook=hook, verbose=args.verbose)

        mnist_trainset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        dataset = sy.BaseDataset(
            data=mnist_trainset.train_data,
            targets=mnist_trainset.train_labels,
            transform=mnist_trainset.transform,
        )
        # Note, that using virtual workers, all workers have all numbers.
        alice.add_dataset(dataset, key="mnist")
        bob.add_dataset(dataset, key="mnist")
        charlie.add_dataset(dataset, key="mnist")
    else:
        kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
        alice = sy.workers.WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
        bob = sy.workers.WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
        charlie = sy.workers.WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)

    worker_instances = [alice, bob, charlie]

    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )

    model = Net().to(device)

    (data, target) = test_loader.__iter__().next()
    traced_model = torch.jit.trace(model, data)

    for epoch in range(1, args.epochs + 1):
        logger.info("Starting epoch %s/%s", epoch, args.epochs)

        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=worker,
                    traced_model=traced_model,
                    batch_size=args.batch_size,
                    curr_epoch=epoch,
                    max_nr_batches=args.federate_after_n_batches,
                )
                for worker in worker_instances
            ]
        )
        models = {}
        loss_values = {}
        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss

        traced_model = utils.federated_avg(models)
        data, target = test_loader.__iter__().next()
        pred_test = traced_model(data)
        loss = loss_fn(output=pred_test, target=target)
        logger.info("Test dataset: Loss: %s, accuracy = %s", loss, accuracy(pred_test, target))

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    # FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
    # LOG_LEVEL = logging.DEBUG
    # logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    asyncio.get_event_loop().run_until_complete(main())
