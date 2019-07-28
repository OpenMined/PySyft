import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

import logging
import argparse
import sys
import asyncio
import numpy as np

import syft as sy
from syft import workers
from syft.frameworks.torch.federated import utils

logger = logging.getLogger(__name__)

LOG_INTERVAL = 25


# Loss function
@torch.jit.script
def loss_fn(pred, target):
    return F.nll_loss(input=pred, target=target)


# Model
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
    parser.add_argument(
        "--training_rounds", type=int, default=40, help="number of federated learning rounds"
    )
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


async def fit_model_on_worker(
    worker: workers.WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    batch_size: int,
    curr_round: int,
    max_nr_batches: int,
    lr: float,
):
    """Send the model to the worker and fit the model on the worker's training data.

    Args:
        worker: Remote location, where the model shall be trained.
        traced_model: Model which shall be trained.
        batch_size: Batch size of each training step.
        curr_round: Index of the current training round (for logging purposes).
        max_nr_batches: If > 0, training on worker will stop at min(max_nr_batches, nr_available_batches).
        lr: Learning rate of each training step.

    Returns:
        A tuple containing:
            * worker_id: Union[int, str], id of the worker.
            * improved model: torch.jit.ScriptModule, model after training at the worker.
            * loss: Loss on last training batch, torch.tensor.
    """
    train_config = sy.TrainConfig(
        model=traced_model,
        loss_fn=loss_fn,
        batch_size=batch_size,
        shuffle=True,
        max_nr_batches=max_nr_batches,
        epochs=1,
        optimizer="SGD",
        optimizer_args={"lr": 0.1},
    )
    train_config.send(worker)
    logger.info("Training round %s, calling fit on worker: %s", curr_round, worker.id)
    loss = await worker.async_fit(dataset_key="mnist", return_ids=[0])
    logger.info("Training round: %s, worker: %s, avg_loss: %s", curr_round, worker.id, loss.mean())
    model = train_config.model_ptr.get().obj
    return worker.id, model, loss


def evaluate_model_on_worker(
    model_identifier, worker, dataset_key, model, nr_bins, batch_size, print_target_hist=False
):
    model.eval()

    # Create and send train config
    train_config = sy.TrainConfig(
        batch_size=batch_size, model=model, loss_fn=loss_fn, optimizer_args=None, epochs=1
    )

    train_config.send(worker)

    result = worker.evaluate(
        dataset_key=dataset_key, return_histograms=True, nr_bins=nr_bins, return_loss=True
    )
    test_loss, correct, len_dataset, hist_pred, hist_target = result

    if print_target_hist:
        logger.info("Target histogram: %s", hist_target)
    logger.info("%s: Prediction hist.: %s", model_identifier, hist_pred)
    logger.info(
        "%s: Percentage numbers 0-3: %s", model_identifier, sum(hist_pred[0:4]) / len_dataset
    )
    logger.info(
        "%s: Percentage numbers 4-6: %s", model_identifier, sum(hist_pred[4:7]) / len_dataset
    )
    logger.info(
        "%s: Percentage numbers 7-9: %s", model_identifier, sum(hist_pred[7:10]) / len_dataset
    )

    logger.info(
        "%s: Test set: Average loss: %s, Accuracy: %s/%s (%s)",
        model_identifier,
        "{:.4f}".format(test_loss),
        correct,
        len_dataset,
        "{:.2f}".format(100.0 * correct / len_dataset),
    )


async def main():
    args = define_and_get_arguments()

    hook = sy.TorchHook(torch)

    kwargs_websocket = {"hook": hook, "verbose": args.verbose, "host": "0.0.0.0"}
    alice = workers.WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
    bob = workers.WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
    charlie = workers.WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)
    testing = workers.WebsocketClientWorker(id="testing", port=8780, **kwargs_websocket)

    worker_instances = [alice, bob, charlie]

    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
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

    for curr_round in range(1, args.training_rounds + 1):
        logger.info("Starting training round %s/%s", curr_round, args.training_rounds)

        results = await asyncio.gather(
            *[
                fit_model_on_worker(
                    worker=worker,
                    traced_model=traced_model,
                    batch_size=args.batch_size,
                    curr_round=curr_round,
                    max_nr_batches=args.federate_after_n_batches,
                    lr=learning_rate,
                )
                for worker in worker_instances
            ]
        )
        models = {}
        loss_values = {}

        test_models = curr_round % 10 == 1 or curr_round == args.training_rounds
        if test_models:
            np.set_printoptions(formatter={"float": "{: .0f}".format})
            for worker_id, worker_model, _ in results:
                evaluate_model_on_worker(
                    model_identifier=worker_id,
                    worker=testing,
                    dataset_key="mnist_testing",
                    model=worker_model,
                    nr_bins=10,
                    batch_size=128,
                    print_target_hist=False,
                )

        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss

        traced_model = utils.federated_avg(models)
        if test_models:
            evaluate_model_on_worker(
                model_identifier="Federated model",
                worker=testing,
                dataset_key="mnist_testing",
                model=traced_model,
                nr_bins=10,
                batch_size=128,
                print_target_hist=True,
            )

        # decay learning rate
        learning_rate = max(0.98 * learning_rate, args.lr * 0.01)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    # Logging setup
    logger = logging.getLogger("run_websocket_server")
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d, p:%(process)d) - %(message)s"
    logging.basicConfig(format=FORMAT)
    logger.setLevel(level=logging.DEBUG)

    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())

    # Run main
    asyncio.get_event_loop().run_until_complete(main())
