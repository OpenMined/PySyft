import math
import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset:
    """
    This is a base class to be used for manipulating a dataset. This is composed
    of a .data attribute for inputs and a .targets one for labels. It is to
    be used like the MNIST Dataset object, and is useful to avoid handling
    the two inputs and label tensors separately.

    Args:

        data[list,torch tensors]: the data points
        targets: Corresponding labels of the data points
        transform: Function to transform the datapoints

    """

    def __init__(self, data, targets, transform=None):

        self.data = data
        self.targets = targets
        self.transform_ = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        """
        Args:

            index[integer]: index of item to get

        Returns:

            data: Data points corresponding to the given index
            targets: Targets correspoding to given datapoint
        """
        data_elem = self.data[index]
        if self.transform_ is not None:
            # TODO: avoid passing through numpy domain
            data_elem = torch.tensor(self.transform_(data_elem.numpy()))

        return data_elem, self.targets[index]

    def transform(self, transform):

        """
         Allows a transform to be applied on given dataset.
         Args:

            transform: The transform to be applied on the data
        """

        # Transforms cannot be applied to Pointer, Fixed Precision or Float Precision tensors.
        if type(self.data) == torch.Tensor:

            self.data = transform(self.data)

        else:

            raise TypeError("Transforms can be applied only on torch tensors")

    def send(self, worker):
        """
        Args:

            worker[worker class]: worker to which the data must be sent

        Returns:

            self: Return the object instance with data sent to corresponding worker

        """

        self.data.send_(worker)
        self.targets.send_(worker)
        return self

    def get(self):
        """
        Gets the data back from respective workers.
        """

        self.data.get_()
        self.targets.get_()
        return self

    def fix_prec(self, *args, **kwargs):
        """
            Converts data of BaseDataset into fixed precision
        """
        self.data.fix_prec_(*args, **kwargs)
        self.targets.fix_prec_(*args, **kwargs)
        return self

    fix_precision = fix_prec

    def float_prec(self, *args, **kwargs):
        """
            Converts data of BaseDataset into float precision
        """
        self.data.float_prec_(*args, **kwargs)
        self.targets.float_prec_(*args, **kwargs)
        return self

    float_precision = float_prec

    def share(self, *args, **kwargs):
        """
            Share the data with the respective workers
        """
        self.data.share_(*args, **kwargs)
        self.targets.share_(*args, **kwargs)
        return self

    @property
    def location(self):
        """
            Get location of the data
        """
        return self.data.location


def dataset_federate(dataset, workers):
    """
    Add a method to easily transform a torch.Dataset or a sy.BaseDataset
    into a sy.FederatedDataset. The dataset given is split in len(workers)
    part and sent to each workers
    """
    logger.info("Scanning and sending data to {}...".format(", ".join([w.id for w in workers])))

    # take ceil to have exactly len(workers) sets after splitting
    data_size = math.ceil(len(dataset) / len(workers))

    # Fix for old versions of torchvision
    if not hasattr(dataset, "data"):
        if hasattr(dataset, "train_data"):
            dataset.data = dataset.train_data
        elif hasattr(dataset, "test_data"):
            dataset.data = dataset.test_data
        else:
            raise AttributeError("Could not find inputs in dataset")
    if not hasattr(dataset, "targets"):
        if hasattr(dataset, "train_labels"):
            dataset.targets = dataset.train_labels
        elif hasattr(dataset, "test_labels"):
            dataset.targets = dataset.test_labels
        else:
            raise AttributeError("Could not find targets in dataset")

    datasets = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=data_size)
    for dataset_idx, (data, targets) in enumerate(data_loader):
        worker = workers[dataset_idx % len(workers)]
        logger.debug("Sending data to worker %s", worker.id)
        data = data.send(worker)
        targets = targets.send(worker)
        datasets.append(BaseDataset(data, targets))  # .send(worker)

    logger.debug("Done!")
    return FederatedDataset(datasets)


Dataset.federate = dataset_federate
BaseDataset.federate = dataset_federate


class FederatedDataset:
    def __init__(self, datasets):
        """This class takes a list of datasets, each of which is supposed
        to be already sent to a remote worker (they have a location), and
        acts like a dictionary based on the worker ids.
        It serves like an input for the FederatedDataLoader.
        Args:
            datasets (list): list of remote Datasets
        """
        self.datasets = {}
        for dataset in datasets:
            worker_id = dataset.data.location.id
            self.datasets[worker_id] = dataset

        # Check that data and targets for a worker are consistent
        for worker_id in self.workers:
            dataset = self.datasets[worker_id]
            assert len(dataset.data) == len(
                dataset.targets
            ), "On each worker, the input and target must have the same number of rows."

    @property
    def workers(self):
        """
           Returns: list of workers
        """

        return list(self.datasets.keys())

    def __getitem__(self, worker_id):
        """
           Args:
                   worker_id[str,int]: ID of respective worker

           Returns: Get Datasets from the respective worker
        """

        return self.datasets[worker_id]

    def __len__(self):

        return sum([len(dataset) for w, dataset in self.datasets.items()])

    def __repr__(self):

        fmt_str = "FederatedDataset\n"
        fmt_str += "    Distributed accross: {}\n".format(", ".join(str(x) for x in self.workers))
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        return fmt_str
