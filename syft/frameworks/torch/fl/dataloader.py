import torch
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler
from torch._six import string_classes, int_classes, container_abcs

import logging
import math

numpy_type_map = {
    "float64": torch.DoubleTensor,
    "float32": torch.FloatTensor,
    "float16": torch.HalfTensor,
    "int64": torch.LongTensor,
    "int32": torch.IntTensor,
    "int16": torch.ShortTensor,
    "int8": torch.CharTensor,
    "uint8": torch.ByteTensor,
}


def default_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):  # pragma: no cover
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith("float") else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):  # pragma: no cover
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):  # pragma: no cover
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):  # pragma: no cover
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):  # pragma: no cover
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):  # pragma: no cover
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class _DataLoaderIter(object):
    """Iterates once over the DataLoader's dataset, as specified by the samplers"""

    def __init__(self, loader, worker_idx):
        self.loader = loader
        self.federated_dataset = loader.federated_dataset

        # Assign the first worker to invoke
        self.worker_idx = worker_idx
        # List workers in a dict
        self.workers = {idx: worker for idx, worker in enumerate(loader.workers)}

        # The function used to stack all samples together
        self.collate_fn = loader.collate_fn

        # Create a sample iterator for each worker
        self.sample_iter = {
            worker: iter(batch_sampler) for worker, batch_sampler in loader.batch_samplers.items()
        }

    def __len__(self):
        return len(self.federated_dataset)

    def _get_batch(self):
        # If all workers have been used, end the iterator
        if len(self.workers) == 0:
            self.stop()

        worker = self.workers[self.worker_idx]

        try:
            indices = next(self.sample_iter[worker])
            batch = self.collate_fn([self.federated_dataset[worker][i] for i in indices])
            return batch
        # All the data for this worker has been used
        except StopIteration:
            # Forget this worker
            del self.workers[self.worker_idx]
            # Find another worker which is not busy
            worker_busy_ids = [it.worker_idx for it in self.loader.iterators]
            for idx in self.workers.keys():
                if idx not in worker_busy_ids:
                    self.worker_idx = idx
                    return self._get_batch()

            # If nothing is found, stop the iterator
            self.stop()

    def __next__(self):
        batch = self._get_batch()
        return batch

    def __iter__(self):
        return self

    def stop(self):
        self.worker_idx = -1
        raise StopIteration


class _DataLoaderOneWorkerIter(object):
    """Iterates once over the worker's dataset, as specified by its sampler"""

    def __init__(self, loader, worker_idx):
        self.loader = loader
        self.federated_dataset = loader.federated_dataset

        # Assign the worker to invoke
        self.worker = loader.workers[worker_idx]

        # The function used to stack all samples together
        self.collate_fn = loader.collate_fn

        # Create a sample iterator for each worker
        self.sample_iter = iter(loader.batch_samplers[self.worker])

    def _get_batch(self):
        # If all workers have been used, end the iterator
        if not self.worker:
            self.stop()

        try:
            indices = next(self.sample_iter)
            batch = self.collate_fn([self.federated_dataset[self.worker][i] for i in indices])
            return batch
        # All the data for this worker has been used
        except StopIteration:
            # If nothing is found, stop the iterator
            self.stop()

    # TODO: implement a length function. It should return the number of elements
    #       of the federated dataset that are located at this worker
    # def __len__(self):
    #    return len(self.federated_dataset)

    def __next__(self):
        return self._get_batch()

    def __iter__(self):
        return self

    def stop(self):
        self.worker = None
        raise StopIteration


class FederatedDataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single or several iterators over the dataset.

    Arguments:
        federated_dataset (FederatedDataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        num_iterators (int): number of workers from which to retrieve data in parallel.
            num_iterators <= len(federated_dataset.workers) - 1
            the effect is to retrieve num_iterators epochs of data but at each step data
            from num_iterators distinct workers is returned.
        iter_per_worker (bool): if set to true, __next__() will return a dictionary
            containing one batch per worker
    """

    __initialized = False

    def __init__(
        self,
        federated_dataset,
        batch_size=8,
        shuffle=False,
        num_iterators=1,
        drop_last=False,
        collate_fn=default_collate,
        iter_per_worker=False,
        **kwargs,
    ):
        if len(kwargs) > 0:
            options = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
            logging.warning(f"The following options are not supported: {options}")

        try:
            self.workers = federated_dataset.workers
        except AttributeError:
            raise Exception(
                "Your dataset is not a FederatedDataset, please use "
                "torch.utils.data.DataLoader instead."
            )

        self.federated_dataset = federated_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.iter_class = _DataLoaderOneWorkerIter if iter_per_worker else _DataLoaderIter

        # Build a batch sampler per worker
        self.batch_samplers = {}
        for worker in self.workers:
            data_range = range(len(federated_dataset[worker]))
            if shuffle:
                sampler = RandomSampler(data_range)
            else:
                sampler = SequentialSampler(data_range)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
            self.batch_samplers[worker] = batch_sampler

        if iter_per_worker:
            self.num_iterators = len(self.workers)
        else:
            # You can't have more iterators than n - 1 workers, because you always
            # need a worker idle in the worker switch process made by iterators
            if len(self.workers) == 1:
                self.num_iterators = 1
            else:
                self.num_iterators = min(num_iterators, len(self.workers) - 1)

    def __iter__(self):
        self.iterators = []
        for idx in range(self.num_iterators):
            self.iterators.append(self.iter_class(self, worker_idx=idx))
        return self

    def __next__(self):
        if self.num_iterators > 1:
            batches = {}
            for iterator in self.iterators:
                data, target = next(iterator)
                batches[data.location] = (data, target)
            return batches
        else:
            iterator = self.iterators[0]
            data, target = next(iterator)
            return data, target

    def __len__(self):
        length = len(self.federated_dataset) / self.batch_size
        if self.drop_last:
            return int(length)
        else:
            return math.ceil(length)
