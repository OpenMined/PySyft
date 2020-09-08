import math
import logging
from syft.generic.abstract.sendable import AbstractSendable
from syft.workers.base import BaseWorker
from syft.generic.pointers.pointer_dataset import PointerDataset
from syft_proto.frameworks.torch.fl.v1.dataset_pb2 import BaseDataset as BaseDatasetPB

import torch
from torch.utils.data import Dataset
import syft

logger = logging.getLogger(__name__)


class BaseDataset(AbstractSendable):
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

    def __init__(self, data, targets, transform=None, owner=None, **kwargs):
        if owner is None:
            owner = syft.framework.hook.local_worker
        super().__init__(owner=owner, **kwargs)
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
        """Allows a transform to be applied on given dataset.

        Args:
            transform: The transform to be applied on the data
        """

        # Transforms cannot be applied to Pointer, Fixed Precision or Float Precision tensors.
        if type(self.data) == torch.Tensor:

            self.data = transform(self.data)

        else:

            raise TypeError("Transforms can be applied only on torch tensors")

    def send(self, location: BaseWorker):
        ptr = self.owner.send(self, workers=location)
        return ptr

    def get(self):
        """
        Gets the data back from respective workers.
        """

        self.data.get_()
        self.targets.get_()
        return self

    def get_data(self):
        return self.data

    def get_targets(self):
        return self.targets

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

    def create_pointer(
        self, owner, garbage_collect_data, location=None, id_at_location=None, **kwargs
    ):
        """creats a pointer to the self dataset"""
        if owner is None:
            owner = self.owner

        if location is None:
            location = self.owner

        owner = self.owner.get_worker(owner)
        location = self.owner.get_worker(location)

        return PointerDataset(
            owner=owner,
            location=location,
            id_at_location=id_at_location or self.id,
            garbage_collect_data=garbage_collect_data,
            tags=self.tags,
            description=self.description,
        )

    def __repr__(self):

        fmt_str = "BaseDataset\n"
        fmt_str += f"\tData: {self.data}\n"
        fmt_str += f"\ttargets: {self.targets}"

        if self.tags is not None and len(self.tags):
            fmt_str += "\n\tTags: "
            for tag in self.tags:
                fmt_str += str(tag) + " "

        if self.description is not None:
            fmt_str += "\n\tDescription: " + str(self.description).split("\n")[0] + "..."

        return fmt_str

    @property
    def location(self):
        """
        Get location of the data
        """
        return self.data.location

    @staticmethod
    def simplify(worker, dataset: "BaseDataset") -> tuple:
        chain = None
        if hasattr(dataset, "child"):
            chain = syft.serde.msgpack.serde._simplify(worker, dataset.child)
        return (
            syft.serde.msgpack.serde._simplify(worker, dataset.data),
            syft.serde.msgpack.serde._simplify(worker, dataset.targets),
            dataset.id,
            syft.serde.msgpack.serde._simplify(worker, dataset.tags),
            syft.serde.msgpack.serde._simplify(worker, dataset.description),
            chain,
        )

    @staticmethod
    def detail(worker, dataset_tuple: tuple) -> "BaseDataset":
        data, targets, id, tags, description, chain = dataset_tuple
        dataset = BaseDataset(
            syft.serde.msgpack.serde._detail(worker, data),
            syft.serde.msgpack.serde._detail(worker, targets),
            owner=worker,
            id=id,
            tags=syft.serde.msgpack.serde._detail(worker, tags),
            description=syft.serde.msgpack.serde._detail(worker, description),
        )
        if chain is not None:
            chain = syft.serde.msgpack.serde._detail(worker, chain)
            dataset.child = chain
        return dataset

    @staticmethod
    def bufferize(worker, dataset):
        """
        This method serializes a BaseDataset into a BaseDatasetPB.

        Args:
            dataset (BaseDataset): input BaseDataset to be serialized.

        Returns:
            proto_dataset (BaseDatasetPB): serialized BaseDataset.
        """
        proto_dataset = BaseDatasetPB()
        proto_dataset.data.CopyFrom(syft.serde.protobuf.serde._bufferize(worker, dataset.data))
        proto_dataset.targets.CopyFrom(
            syft.serde.protobuf.serde._bufferize(worker, dataset.targets)
        )
        syft.serde.protobuf.proto.set_protobuf_id(proto_dataset.id, dataset.id)
        for tag in dataset.tags:
            proto_dataset.tags.append(tag)

        if dataset.child:
            proto_dataset.child.CopyFrom(dataset.child)

        proto_dataset.description = dataset.description
        return proto_dataset

    @staticmethod
    def unbufferize(worker, proto_dataset):
        """
        This method deserializes BaseDatasetPB into a BaseDataset.

        Args:
            proto_dataset (BaseDatasetPB): input serialized BaseDatasetPB.

        Returns:
             BaseDataset: deserialized BaseDatasetPB.
        """
        data = syft.serde.protobuf.serde._unbufferize(worker, proto_dataset.data)
        targets = syft.serde.protobuf.serde._unbufferize(worker, proto_dataset.targets)
        dataset_id = syft.serde.protobuf.proto.get_protobuf_id(proto_dataset.id)
        child = None
        if proto_dataset.HasField("child"):
            child = syft.serde.protobuf.serde._unbufferize(worker, proto_dataset.child)
        return BaseDataset(
            data=data,
            targets=targets,
            id=dataset_id,
            tags=set(proto_dataset.tags),
            description=proto_dataset.description,
            child=child,
        )

    @staticmethod
    def get_protobuf_schema():
        """
        This method returns the protobuf schema used for BaseDataset.

        Returns:
           Protobuf schema for BaseDataset.
        """
        return BaseDatasetPB


def dataset_federate(dataset, workers):
    """
    Add a method to easily transform a torch.Dataset or a sy.BaseDataset
    into a sy.FederatedDataset. The dataset given is split in len(workers)
    part and sent to each workers
    """
    logger.info(f"Scanning and sending data to {', '.join([w.id for w in workers])}...")

    # take ceil to have exactly len(workers) sets after splitting
    data_size = math.ceil(len(dataset) / len(workers))

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
            dataset.federated = True

        # Check that data and targets for a worker are consistent
        """for worker_id in self.workers:
            dataset = self.datasets[worker_id]
            assert (
                dataset.data.shape == dataset.targets.shape
            ), "On each worker, the input and target must have the same number of rows.""" ""

    @property
    def workers(self):
        """
        Returns: list of workers
        """

        return list(self.datasets.keys())

    def get_dataset(self, worker):
        self[worker].federated = False
        dataset = self[worker].get()
        del self.datasets[worker]
        return dataset

    def __getitem__(self, worker_id):
        """
        Args:
            worker_id[str,int]: ID of respective worker

        Returns:
            Get Datasets from the respective worker
        """

        return self.datasets[worker_id]

    def __len__(self):

        return sum(len(dataset) for dataset in self.datasets.values())

    def __repr__(self):

        fmt_str = "FederatedDataset\n"
        fmt_str += f"    Distributed accross: {', '.join(str(x) for x in self.workers)}\n"
        fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        return fmt_str
