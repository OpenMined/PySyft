import argparse

import torch

torch.set_num_threads(1)

import syft as sy
from syft.serde.compression import NO_COMPRESSION

sy.serde.compression.default_compress_scheme = NO_COMPRESSION

from examples.ariann.procedure import train, test
from examples.ariann.data import get_data_loaders, get_number_classes
from examples.ariann.models import get_model
from examples.ariann.preprocess import build_prepocessing


def run_inference(args):
    print("Running inference speed test on", args.model, args.dataset)

    hook = sy.TorchHook(torch)
    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

    workers = [alice, bob]
    sy.local_worker.clients = workers

    kwargs = dict(crypto_provider=crypto_provider, protocol=args.protocol)

    if args.preprocess:
        build_prepocessing(args.model, args.dataset, workers, args)

    private_train_loader, private_test_loader = get_data_loaders(
        workers, args, kwargs, private=True
    )
    # public_train_loader, public_test_loader = get_data_loaders(workers, args, kwargs, private=False)

    model = get_model(args.model, out_features=get_number_classes(args.dataset))

    model.fix_precision(precision_fractional=args.precision_fractional, dtype=args.dtype).share(
        *workers, **kwargs
    )
    test_time, accuracy = test(args, model, private_test_loader)

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
        help="Model to test on inference",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="DataSet to use",
    )

    parser.add_argument("--preprocess", help="Preprocess data or not", action="store_true")

    cmd_args = parser.parse_args()

    class Arguments:
        model = cmd_args.model.lower()
        dataset = cmd_args.dataset.lower()
        preprocess = cmd_args.preprocess

        epochs = 1

        VAL = 1
        n_train_items = VAL
        n_test_items = VAL

        batch_size = VAL
        test_batch_size = VAL

        dtype = "long"
        protocol = "fss"
        precision_fractional = 4
        lr = 0.1
        log_interval = 40

    args = Arguments()

    run_inference(args)
