import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import logging
import argparse
import sys
import asyncio
import numpy as np

FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
LOG_LEVEL = logging.INFO
logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

import syft as sy

from syft import workers
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
        "--test_batch_size", type=int, default=128, help="batch size used for the test data"
    )
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=10,
        help="number of training steps performed on each remote worker before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will be started in verbose mode",
    )

    args = parser.parse_args(args=args)
    return args


def accuracy(pred_softmax, target):
    nr_elems = len(target)
    pred = pred_softmax.argmax(dim=1)
    logger.debug("predicted: %s", pred)
    logger.debug("target:    %s", target)
    return (pred == target).sum().numpy() / float(nr_elems)


async def fit_model_on_worker(worker, traced_model, batch_size, curr_epoch, max_nr_batches, lr):
    """Send the model to the worker and fit the model on the worker's training data

    Args:
        worker: "workers.WebsocketClientWorker", remote location, where the model shall be trained
        traced_model: torch.jit.ScriptModule, model which shall be trained
        batch_size: int, the batch size of each training step
        curr_epoch: int, index of the currently trained epoch (for logging purposes)
        max_nr_batches: int, if > 0, training on worker will stop at min(max_nr_batches, nr_available_batches)
        lr: learning rate of each training step

    Returns:
        (worker_id, improved model, loss on last training batch)
        worker_id: Union[int, str], id of the worker
        improved model: torch.jit.ScriptModule, model after training at the worker
        loss on last training batch, torch.tensor

    """
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=max_nr_batches,
        epochs=1,
        lr=lr,
    )
    train_config.send(worker)
    logger.info(
        "Training round %s, calling fit on worker: %s, lr = %s",
        curr_epoch,
        worker.id,
        train_config.lr,
    )
    loss = await worker.fit(dataset_key="mnist", return_ids=[0])
    logger.info("Training round: %s, worker: %s, avg_loss: %s", curr_epoch, worker.id, loss.mean())
    model = train_config.model_ptr.get().obj
    return worker.id, model, loss


def __evaluate_models_on_test_data(test_loader, results):
    data, target = test_loader.__iter__().next()
    logger.info("Testing individual models on test data")
    # logger.info("Target:    %s", target)
    hist, bin_edges = np.histogram(target, bins=10, range=(0, 10))
    logger.info("Target hist: %s", hist)
    for worker_id, worker_model, worker_loss in results:
        pred_test = worker_model(data)
        loss = loss_fn(output=pred_test, target=target)
        logger.info(
            "Worker %s: Loss: %s, accuracy = %s", worker_id, loss, accuracy(pred_test, target)
        )
        pred = pred_test.argmax(dim=1)
        hist, bin_edges = np.histogram(pred, bins=10, range=(0, 10))
        logger.info("Worker %s: Predicted hist: %s", worker_id, hist)
        # logger.info("Predicted: %s", pred)


def evaluate_models_on_test_data(test_loader, results):
    np.set_printoptions(formatter={"float": "{: .0f}".format})
    for worker_id, worker_model, worker_loss in results:
        evaluate_model(worker_id, worker_model, "cpu", test_loader)


def evaluate_model(worker_id, model, device, test_loader, print_target_hist=False):
    model.eval()
    test_loss = 0
    correct = 0
    hist_target = np.zeros(10)
    hist_pred = np.zeros(10)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            hist, bin_edges = np.histogram(target, bins=10, range=(0, 10))
            hist_target += hist
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            hist, bin_edges = np.histogram(pred, bins=10, range=(0, 10))
            hist_pred += hist
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if print_target_hist:
        logger.info("Target hist    : %s", hist_target)
    logger.info("Prediction hist: %s", hist_pred)

    logger.info(
        "%s: Test set: Average loss: %s, Accuracy: %s/%s (%s)",
        worker_id,
        "{:.4f}".format(test_loss),
        correct,
        len(test_loader.dataset),
        "{:.0f}".format(100.0 * correct / len(test_loader.dataset)),
    )


async def main():
    args = define_and_get_arguments()

    hook = sy.TorchHook(torch)

    kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
    alice = workers.WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
    bob = workers.WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
    charlie = workers.WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)

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
        shuffle=False,
        drop_last=False,
        **kwargs,
    )

    model = Net().to(device)

    (data, target) = test_loader.__iter__().next()
    traced_model = torch.jit.trace(model, data)
    learning_rate = args.lr

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
                    lr=learning_rate,
                )
                for worker in worker_instances
            ]
        )
        models = {}
        loss_values = {}

        test_models = epoch % 10 == 1 or epoch == args.epochs
        if test_models:
            evaluate_models_on_test_data(test_loader, results)

        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss

        traced_model = utils.federated_avg(models)
        if test_models:
            evaluate_model("Federated model", traced_model, "cpu", test_loader)

        # decay learning rate
        learning_rate = max(0.98 * learning_rate, args.lr * 0.01)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":

    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    asyncio.get_event_loop().run_until_complete(main())
