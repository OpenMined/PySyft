import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import logging
import argparse
import sys

FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
LOG_LEVEL = logging.DEBUG
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
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs to train")
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


def main():
    args = define_and_get_arguments()
    # args.use_virtual = True

    hook = sy.TorchHook(torch)
    me = hook.local_worker

    if args.use_virtual:
        alice = sy.workers.VirtualWorker(id="alice", hook=hook, verbose=args.verbose)
        bob = sy.workers.VirtualWorker(id="bob", hook=hook, verbose=args.verbose)
        charlie = sy.workers.VirtualWorker(id="charlie", hook=hook, verbose=args.verbose)
        # train_loader = torch.utils.data.DataLoader(
        #    datasets.MNIST(
        #        "../data",
        #        train=True,
        #        transform=transforms.Compose(
        #            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        #        ),
        #    ),
        #    batch_size=args.#batch_size,
        #    shuffle=True,
        # )
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
        alice.add_dataset(dataset, key="mnist")
        bob.add_dataset(dataset, key="mnist")
        charlie.add_dataset(dataset, key="mnist")
    else:
        kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
        alice = sy.workers.WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
        bob = sy.workers.WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
        charlie = sy.workers.WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)

    # worker_instances = [alice, bob, charlie]
    worker_instances = [charlie, alice, bob]

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

    # dataset = datasets.MNIST(
    #        "../data",
    #        train=False,
    #        transform=transforms.Compose(
    #            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #        ),
    #    )

    model = Net().to(device)
    #    (data, target) = test_loader.__iter__().next()
    # sampler = torch.utils.data.SequentialSampler(len(dataset))

    # train_config = sy.TrainConfig(batch_size=32, shuffle=True, max_nr_batches=10)
    # batch_sampler = torch.utils.data.BatchSampler(sampler, train_config.batch_size, drop_last=True)
    (data, target) = test_loader.__iter__().next()
    traced_model = torch.jit.trace(model, data)

    # model_with_id = pointers.ObjectWrapper(id=sy.ID_PROVIDER.pop(), obj=traced_model)
    # loss_fn_with_id = pointers.ObjectWrapper(id=sy.ID_PROVIDER.pop(), obj=loss_fn)

    for epoch in range(1, args.epochs + 1):
        logger.info("Starting epoch %s/%s", epoch, args.epochs)

        models = {}
        loss_values = {}
        for worker in worker_instances:
            # model_ptr = me.send(model_with_id, worker)
            # loss_fn_ptr = me.send(loss_fn_with_id, worker)

            # Create and send train config
            train_config = sy.TrainConfig(
                batch_size=args.batch_size, shuffle=True, max_nr_batches=50, epochs=1
            )
            train_config.send(worker, traced_model=traced_model, traced_loss_fn=loss_fn)
            # pred = train_config.model_ptr(data)
            # shape = data.shape
            # logger.debug("Training round %s, calling fit on worker: %s", epoch, worker.id)
            loss = worker.fit(dataset="mnist", return_ids=[0])
            logger.debug(
                "Training round: %s, worker: %s, avg_loss: %s", epoch, worker.id, loss.mean()
            )
            # logger.debug("Worker state: %s", worker)
            # logger.debug("Worker objects: \n%s", worker.list_objects_remote())
            # logger.debug("loss: mean %s, max %s, min %s", loss.mean(), loss.max(), loss.min())
            if not torch.isnan(loss).any():
                models[worker.id] = train_config.model_ptr.get().obj
            else:
                if worker.id in models:
                    del models[worker.id]
            loss_values[worker.id] = loss
            # logger.debug("Remote objects: %s", worker.list_objects_remote())
            # train_config.get(worker)
            # pred_test = traced_model(data)
            # loss = loss_fn(output=pred_test, target=target)
            # logger.debug("Traced model should not have changed: loss: %s, accuracy = %s", loss, accuracy(pred_test, target))

        traced_model = utils.federated_avg(models)
        data, target = test_loader.__iter__().next()
        pred_test = traced_model(data)
        loss = loss_fn(output=pred_test, target=target)
        logger.debug("loss: %s, accuracy = %s", loss, accuracy(pred_test, target))

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.DEBUG)
    websockets_logger.addHandler(logging.StreamHandler())

    main()
