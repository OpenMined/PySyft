import time

import torch as th
import syft as sy

# fmt: off
config_zoo = {
    "template": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    },
    "network1-mnist-128": {
        'fss_eq': [128],
        'fss_comp': [16384, 16384, 1280, 11520, 1280],
        'mul': [((128, 128), (128, 128)), ((128, 128), (128, 128)), ((128, 10), (128, 10))],
        'matmul': [((128, 784), (784, 128)), ((128, 128), (128, 128)), ((128, 128), (128, 10))],
    },
    "network2-mnist-128": {
        'conv2d': [((128, 1, 28, 28), (16, 1, 5, 5), (('bias', 'stride', 'padding', 'dilation', 'groups'), (None, 1, (0, 0), (1, 1), 1))), ((128, 16, 12, 12), (16, 16, 5, 5), (('bias', 'stride', 'padding', 'dilation', 'groups'), (None, 1, (0, 0), (1, 1), 1)))] ,
        'fss_comp': [589824, 294912, 294912, 65536, 32768, 32768, 12800, 1280, 11520, 1280] ,
        'mul': [((128, 16, 144, 2), (128, 16, 144, 2)), ((128, 16, 144, 1), (128, 16, 144, 1)), ((128, 16, 12, 12), (128, 16, 12, 12)), ((128, 16, 16, 2), (128, 16, 16, 2)), ((128, 16, 16, 1), (128, 16, 16, 1)), ((128, 16, 4, 4), (128, 16, 4, 4)), ((128, 100), (128, 100)), ((128, 10), (128, 10))] ,
        'matmul': [((128, 256), (256, 100)), ((128, 100), (100, 10))] ,
        'fss_eq': [128] ,
    },
    "lenet-mnist-128": {
        'conv2d': [((128, 1, 28, 28), (20, 1, 5, 5), (('bias', 'stride', 'padding', 'dilation', 'groups'), (None, 1, (0, 0), (1, 1), 1))), ((128, 20, 12, 12), (50, 20, 5, 5), (('bias', 'stride', 'padding', 'dilation', 'groups'), (None, 1, (0, 0), (1, 1), 1)))] ,
        'fss_comp': [737280, 368640, 368640, 204800, 102400, 102400, 64000, 1280, 11520, 1280] ,
        'mul': [((128, 20, 144, 2), (128, 20, 144, 2)), ((128, 20, 144, 1), (128, 20, 144, 1)), ((128, 20, 12, 12), (128, 20, 12, 12)), ((128, 50, 16, 2), (128, 50, 16, 2)), ((128, 50, 16, 1), (128, 50, 16, 1)), ((128, 50, 4, 4), (128, 50, 4, 4)), ((128, 500), (128, 500)), ((128, 10), (128, 10))] ,
        'matmul': [((128, 800), (800, 500)), ((128, 500), (500, 10))] ,
        'fss_eq': [128] ,
    },
    "alexnet-cifar10-128": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    },
    "alexnet-tiny-imagenet-128": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    },
    "vgg16-cifar10-1": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    },
    "vgg16-cifar10-64": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    },
    "vgg16-cifar10-128": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    },
    "vgg16-tiny-imagenet-1": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    },
    "vgg16-tiny-imagenet-16": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    },
    "vgg16-tiny-imagenet-32": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    },
    "resnet18-hymenoptera-1": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    },
    "resnet18-hymenoptera-4": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    },
    "resnet18-hymenoptera-8": {
        'fss_eq': [],
        'fss_comp': [],
        'mul': [],
        'matmul': [],
        'conv2d': []
    }
}
# fmt: on


def build_prepocessing(model, dataset, batch_size, workers, args):
    start_time = time.time()

    try:
        config = config_zoo[f"{model}-{dataset}-{batch_size}"]
    except KeyError:
        print(f"WARNING: No preprocessing found for {model}-{dataset}-{batch_size}")
        return 0

    if args.verbose:
        print("Preprocess")

    for op in ["fss_eq", "fss_comp"]:
        n_instances_list = config[op]
        for n_instances in n_instances_list:
            if args.verbose:
                print(f"{op} n_instances", n_instances)
            sy.local_worker.crypto_store.provide_primitives(
                op=op, kwargs_={}, workers=workers, n_instances=n_instances
            )

    for op in {"mul", "matmul", "conv2d"}:
        try:
            shapes = config[op]
        except KeyError:
            continue

        if args.verbose:
            print(f"{op} shapes", shapes)

        if args.dtype == "int":
            torch_dtype = th.int32
            field = 2 ** 32
        elif args.dtype == "long":
            torch_dtype = th.int64
            field = 2 ** 64
        else:
            raise ValueError(f"Unsupported dtype {args.dtype}")

        if op == "conv2d":
            for left_shape, right_shape, hashable_kwargs_ in shapes:
                keys, values = hashable_kwargs_
                kwargs_ = dict(zip(keys, values))
                sy.local_worker.crypto_store.provide_primitives(
                    op=op,
                    kwargs_=kwargs_,
                    workers=workers,
                    n_instances=1,
                    shapes=(left_shape, right_shape),
                    dtype=args.dtype,
                    torch_dtype=torch_dtype,
                    field=field,
                )
        else:
            sy.local_worker.crypto_store.provide_primitives(
                op=op,
                kwargs_={},
                workers=workers,
                n_instances=1,
                shapes=shapes,
                dtype=args.dtype,
                torch_dtype=torch_dtype,
                field=field,
            )

    preprocess_time = time.time() - start_time
    if args.verbose:
        print(
            "...", preprocess_time, "s", "[time per item=", preprocess_time / args.batch_size, "]"
        )
    else:
        print("Preprocessing time (s):\t", round(preprocess_time / args.batch_size, 4))
    return preprocess_time
