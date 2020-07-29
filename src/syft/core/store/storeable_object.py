from typing import List, Optional
from dataclasses import dataclass

from ...common.id import UID
from ...decorators import syft_decorator
from ..serialization import Serializable


@dataclass(frozen=True)
class StorableObject(Serializable):
    key: UID
    data: Serializable
    description: Optional[str]
    tags: Optional[List[str]]

    @syft_decorator(typechecking=True)
    def get_schema(self):
        pass

    @syft_decorator(typechecking=True)
    def to_protobuf(self):
        pass

    @staticmethod
    @syft_decorator(typechecking=True)
    def from_protobuf(proto):
        pass
