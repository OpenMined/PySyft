# stdlib
from typing import Any

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import torch as th
from torch.utils.data import Dataset

# syft relative
from ...core.plan import Plan
from ...proto.core.remote_dataloader.remote_dataset_pb2 import (
    RemoteDataset as RemoteDataset_PB,
)
from ..common.serde.serializable import bind_protobuf


@bind_protobuf
class RemoteDataset(Dataset):
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
