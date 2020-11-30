import argparse
import os
import signal
import subprocess
import time

import torch
import torch.optim as optim

torch.set_num_threads(1)

import syft as sy
from syft.serde.compression import NO_COMPRESSION
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

sy.serde.compression.default_compress_scheme = NO_COMPRESSION

from examples.ariann.procedure import train, test
from examples.ariann.data import get_data_loaders, get_number_classes
from examples.ariann.models import get_model, load_state_dict
from examples.ariann.preprocess import build_prepocessing


def run(args):
    if args.train:
        print(f"Training over {args.epochs} epochs")
    elif args.test:
        print("Running a full evaluation")
    else:
        print("Running inference speed test")
    print("model:\t\t", args.model)
    print("dataset:\t", args.dataset)
    print("batch_size:\t", args.batch_size)

    hook = sy.TorchHook(torch)

    if args.websockets:
        alice = DataCentricFLClient(hook, "ws://localhost:7600")
        bob = DataCentricFLClient(hook, "ws://localhost:7601")
        crypto_provider = DataCentricFLClient(hook, "ws://localhost:7602")
        my_grid = sy.PrivateGridNetwork(alice, bob, crypto_provider)
        sy.local_worker.object_store.garbage_delay = 1

    else:
        bob = sy.VirtualWorker(hook, id="bob")
        alice = sy.VirtualWorker(hook, id="alice")
        crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    workers = [alice, bob]
    sy.local_worker.clients = workers

    encryption_kwargs = dict(
        workers=workers, crypto_provider=crypto_provider, protocol=args.protocol
    )
    kwargs = dict(
        requires_grad=args.requires_grad,
        precision_fractional=args.precision_fractional,
        dtype=args.dtype,
        **encryption_kwargs,
    )

    if args.preprocess:
        build_prepocessing(args.model, args.dataset, args.batch_size, workers, args)

    private_train_loader, private_test_loader = get_data_loaders(args, kwargs, private=True)
    public_train_loader, public_test_loader = get_data_loaders(args, kwargs, private=False)

    model = get_model(args.model, args.dataset, out_features=get_number_classes(args.dataset))

    if args.test and not args.train:
        load_state_dict(model, args.model, args.dataset)

    model.eval()

    if torch.cuda.is_available():
        sy.cuda_force = True

    if not args.public:
        model.encrypt(**kwargs)
        if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
            model.get()

    if args.train:
        for epoch in range(args.epochs):
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
            if not args.public:
                optimizer = optimizer.fix_precision(
                    precision_fractional=args.precision_fractional, dtype=args.dtype
                )
            train_time = train(args, model, private_train_loader, optimizer, epoch)
            test_time, accuracy = test(args, model, private_test_loader)
    else:
        test_time, accuracy = test(args, model, private_test_loader)
        if not args.test:
            print(
                f"{ 'Online' if args.preprocess else 'Total' } time (s):\t",
                round(test_time / args.batch_size, 4),
            )
        else:
            # Compare with clear text accuracy
            print("Clear text accuracy is:")
            model = get_model(
                args.model, args.dataset, out_features=get_number_classes(args.dataset)
            )
            load_state_dict(model, args.model, args.dataset)
            test(args, model, public_test_loader)

    if args.preprocess:
        missing_items = [len(v) for k, v in sy.preprocessed_material.items()]
        if sum(missing_items) > 0:
            print("MISSING preprocessed material")
            for key, value in sy.preprocessed_material.items():
                print(f"'{key}':", value, ",")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="model to use for inference (network1, network2, lenet, alexnet, vgg16, resnet18)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to use (mnist, cifar10, hymenoptera, tiny-imagenet)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="size of the batch to use",
        default=128,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        help="size of the batch to use",
        default=None,
    )

    parser.add_argument(
        "--preprocess", help="[only for speed test] preprocess data or not", action="store_true"
    )

    parser.add_argument(
        "--fp_only",
        help="Don't secret share values, just convert them to fix precision",
        action="store_true",
    )

    parser.add_argument(
        "--public",
        help="[needs --train] Train without fix precision or secret sharing",
        action="store_true",
    )

    parser.add_argument(
        "--test",
        help="run testing on the complete test dataset",
        action="store_true",
    )

    parser.add_argument(
        "--train",
        help="run training for n epochs",
        action="store_true",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="[needs --train] number of epochs to train on",
        default=15,
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="[needs --train] learning rate of the SGD",
        default=0.01,
    )

    parser.add_argument(
        "--websockets",
        help="use PyGrid nodes instead of a virtual network. (nodes are launched automatically)",
        action="store_true",
    )

    parser.add_argument("--verbose", help="show extra information and metrics", action="store_true")

    parser.add_argument(
        "--log_interval",
        type=int,
        help="[needs --test or --train] log intermediate metrics every n batches",
        default=10,
    )

    parser.add_argument(
        "--comm_info",
        help="Print communication information",
        action="store_true",
    )

    parser.add_argument(
        "--pyarrow_info",
        help="print information about PyArrow usage and failure",
        action="store_true",
    )

    cmd_args = parser.parse_args()

    # Sanity checks

    if cmd_args.test or cmd_args.train:
        assert (
            not cmd_args.preprocess
        ), "Can't preprocess for a full epoch evaluation or training, remove --preprocess"

    if cmd_args.train:
        assert not cmd_args.test, "Can't set --test if you already have --train"

    if cmd_args.fp_only:
        assert not cmd_args.preprocess, "Can't have --preprocess in a fixed precision setting"
        assert not cmd_args.public, "Can't have simultaneously --fp_only and --public"

    if not cmd_args.train:
        assert not cmd_args.public, "--public is used only for training"

    if cmd_args.pyarrow_info:
        sy.pyarrow_info = True

    class Arguments:
        model = cmd_args.model.lower()
        dataset = cmd_args.dataset.lower()
        preprocess = cmd_args.preprocess
        websockets = cmd_args.websockets
        verbose = cmd_args.verbose

        train = cmd_args.train
        n_train_items = -1 if cmd_args.train else cmd_args.batch_size
        test = cmd_args.test or cmd_args.train
        n_test_items = -1 if cmd_args.test or cmd_args.train else cmd_args.batch_size

        batch_size = cmd_args.batch_size
        # Defaults to the train batch_size
        test_batch_size = cmd_args.test_batch_size or cmd_args.batch_size

        log_interval = cmd_args.log_interval
        comm_info = cmd_args.comm_info

        epochs = cmd_args.epochs
        lr = 0.1

        public = cmd_args.public
        fp_only = cmd_args.fp_only
        requires_grad = cmd_args.train
        dtype = "long"
        protocol = "fss"
        precision_fractional = 4

    args = Arguments()

    if args.websockets:
        print("Launching the websocket workers...")

        def kill_processes(worker_processes):
            for worker_process in worker_processes:
                pid = worker_process.pid
                try:
                    os.killpg(os.getpgid(worker_process.pid), signal.SIGTERM)
                    print(f"Process {pid} killed")
                except ProcessLookupError:
                    print(f"COULD NOT KILL PROCESS {pid}")

        worker_processes = [
            subprocess.Popen(
                f"./scripts/launch_{worker}.sh",
                stdout=subprocess.PIPE,
                shell=True,
                preexec_fn=os.setsid,
                executable="/bin/bash",
            )
            for worker in ["alice", "bob", "crypto_provider"]
        ]
        time.sleep(7)
        try:
            print("LAUNCHED", *[p.pid for p in worker_processes])
            run(args)
            kill_processes(worker_processes)
        except Exception as e:
            kill_processes(worker_processes)
            raise e

    else:
        run(args)
