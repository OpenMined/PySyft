# stdlib
from typing import Any

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import torch as th
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# syft relative
from ... import deserialize
from ... import serialize
from ...core.plan import Plan
from ...proto.core.remote_dataloader.remote_dataset_pb2 import (
    RemoteDataLoader as RemoteDataLoader_PB,
)
from ...proto.core.remote_dataloader.remote_dataset_pb2 import (
    RemoteDataset as RemoteDataset_PB,
)
from ..common.serde.serializable import Serializable
from ..common.serde.serializable import bind_protobuf


@bind_protobuf
class RemoteDataset(Dataset, Serializable):
    def __init__(self, dataset_meta: str, creator: Plan = None):  # type: ignore
        """
        Arguments:
            dataset_meta: information about where to get the raw data, for example, a file path, or a directory path
        For now, it's should simply be a .pt file, which stores a Dataset object.
            creator: a Plan object, it should take meta info as input, and return a Dataset object.
        For now, we are not realy using it.
        """
        self.datasetmeta = dataset_meta
        self.creator = creator

    def create_dataset(self) -> None:
        """
        Create the real Dataset object on DO's machine.
        Inside, it will call self.creator(self.datasetmeta).
        But for now, it's just simply calling torch.load on a .pt file.
        """
        self.dataset = th.load(self.datasetmeta)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> th.Tensor:
        return self.dataset[i]

    def _object2proto(self) -> RemoteDataset_PB:
        proto = RemoteDataset_PB()
        proto.meta = self.datasetmeta
        return proto

    @staticmethod
    def _proto2object(proto: Any) -> "RemoteDataset":
        return RemoteDataset(proto.meta)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RemoteDataset_PB


@bind_protobuf
class RemoteDataLoader(Serializable):
    def __init__(self, remote_dataset: RemoteDataset, batch_size: int = 4):
        """
        TODO: now, only batch_size can be passed in by users, and it's used when create DataLoader object in
        self.create_dataloader. In future steps, more auguments should be supported, like shuffle, sampler,
        collate_fn, ...
        """
        self.remote_dataset = remote_dataset
        self.batch_size = batch_size

    def _object2proto(self) -> RemoteDataLoader_PB:
        proto = RemoteDataLoader_PB()
        proto.batch_size = self.batch_size
        proto.remote_dataset.CopyFrom(serialize(self.remote_dataset))
        return proto

    @staticmethod
    def _proto2object(proto: Any) -> "RemoteDataLoader":
        remote_dataset = deserialize(proto.remote_dataset)
        batch_size = proto.batch_size
        return RemoteDataLoader(remote_dataset=remote_dataset, batch_size=batch_size)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RemoteDataLoader_PB

    def create_dataset(self) -> None:
        self.remote_dataset.create_dataset()

    def create_dataloader(self) -> None:
        dataset = getattr(self.remote_dataset, "dataset", None)
        self.dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self) -> "RemoteDataLoader":
        self.dataloader_iterator = iter(self.dataloader)
        return self

    def __next__(self) -> th.Tensor:
        return next(self.dataloader_iterator)
