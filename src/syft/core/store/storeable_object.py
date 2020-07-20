from typing import List, Optional
from dataclasses import dataclass

from ...common.id import UID
from ..serialization import Serializable, SerializationProtocol


@dataclass(frozen=True)
class StorableObject(Serializable):
    key: UID
    data: Serializable
    description: Optional[str]
    tags: Optional[List[str]]

    def get_serialization_schemas(self):
        return {
            SerializationProtocol.ProtoBuffers: None,
            SerializationProtocol.Custom: None,
        }
