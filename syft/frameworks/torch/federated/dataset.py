import math

import torch
from torch.utils.data import Dataset


class BaseDataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def send(self, worker):
        self.data.send_(worker)
        self.targets.send_(worker)
        return self

    def get(self):
        self.data.get_()
        self.targets.get_()
        return self

    @property
    def location(self):
        return self.data.location


def dataset_federate(dataset, workers):
    """Add a method to easily transform a torch.Dataset in a sy.FederatedDataset"""

    print("Scanning and sending data to {}...".format(", ".join([w.id for w in workers])))

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
        data = data.send(worker)
        targets = targets.send(worker)
        datasets.append(BaseDataset(data, targets))  # .send(worker)

    print("Done!")
    return FederatedDataset(datasets)


Dataset.federate = dataset_federate
BaseDataset.federate = dataset_federate


class FederatedDataset:
    def __init__(self, datasets):
        """This class takes two dictionaries of data points, the keys
        of which are worker IDs, and their values are tensors or pointers
        to tensors. One can also provide direct FederatedData elements.
        The class will use these collections of datasets or pointers
        to datasets and form them into one cohesive dataset.
        Args:
            datasets (list)
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
        return list(self.datasets.keys())

    def __getitem__(self, worker_id):
        return self.datasets[worker_id]

    def __len__(self):
        return sum([len(dataset) for w, dataset in self.datasets.items()])

    def __repr__(self):
        fmt_str = "FederatedDataset\n"
        fmt_str += "    Distributed accross: {}\n".format(", ".join(self.workers))
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        return fmt_str
