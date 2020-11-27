import argparse
import os
import signal
import subprocess
import time

import torch

torch.set_num_threads(1)

import syft as sy
from syft.serde.compression import NO_COMPRESSION
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

sy.serde.compression.default_compress_scheme = NO_COMPRESSION

from examples.ariann.procedure import train, test
from examples.ariann.data import get_data_loaders, get_number_classes
from examples.ariann.models import get_model
from examples.ariann.preprocess import build_prepocessing


def run_inference(args):
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

    kwargs = dict(crypto_provider=crypto_provider, protocol=args.protocol)

    if args.preprocess:
        build_prepocessing(args.model, args.dataset, args.batch_size, workers, args)

    private_train_loader, private_test_loader = get_data_loaders(
        workers, args, kwargs, private=True
    )
    # public_train_loader, public_test_loader = get_data_loaders(workers, args, kwargs, private=False)

    model = get_model(args.model, args.dataset, out_features=get_number_classes(args.dataset))
    model.eval()

    model.encrypt(
        protocol=args.protocol,
        workers=workers,
        crypto_provider=crypto_provider,
        precision_fractional=args.precision_fractional,
        dtype=args.dtype,
    )

    test_time, accuracy = test(args, model, private_test_loader)

    print(
        f"{ 'Online' if args.preprocess else 'Total' } time (s):\t",
        round(test_time / args.batch_size, 4),
    )

    if args.preprocess:
        missing_items = [len(v) for k, v in sy.preprocessed_material.items()]
        if sum(missing_items) > 0:
            print("MISSING preprocessed material")
            print(sy.preprocessed_material)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="model to use for inference (network1, network2, lenet, alexnet, vgg16, resnet18)",
    )

    parser.add_argument(
        "--dataset", type=str, help="dataset to use (mnist, cifar10, hymenoptera, tiny-imagenet)",
    )

    parser.add_argument(
        "--batch_size", type=int, help="size of the batch to use", default=128,
    )

    parser.add_argument("--preprocess", help="preprocess data or not", action="store_true")

    parser.add_argument(
        "--websockets",
        help="use PyGrid nodes instead of a virtual network. (nodes are launched automatically)",
        action="store_true",
    )

    parser.add_argument("--verbose", help="preprocess data or not", action="store_true")

    cmd_args = parser.parse_args()

    class Arguments:
        model = cmd_args.model.lower()
        dataset = cmd_args.dataset.lower()
        preprocess = cmd_args.preprocess
        websockets = cmd_args.websockets
        verbose = cmd_args.verbose

        epochs = 1

        VAL = cmd_args.batch_size
        n_train_items = VAL
        n_test_items = VAL

        batch_size = cmd_args.batch_size
        test_batch_size = VAL

        dtype = "long"
        protocol = "fss"
        precision_fractional = 4
        lr = 0.1
        log_interval = 40

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
            run_inference(args)
            kill_processes(worker_processes)
        except Exception as e:
            kill_processes(worker_processes)
            raise e

    else:
        run_inference(args)
