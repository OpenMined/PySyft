from typing import List, Optional
from dataclasses import dataclass
from ...common.id import UID
from ..serialization import Serializable, SerializationProtocol

@dataclass(frozen=True)
class StorableObject(Serializable):
    key: UID
    data: bin
    description: Optional[str]
    tags: Optional[List[str]]

    @staticmethod
    def from_serializable(key: UID, data: Serializable, description: Optional[str],
                          tags: Optional[List[str]]) -> "StorableObject":
        return StorableObject(key=key, data=data.serialize(), description=description, tags=tags)

    @staticmethod
    def serialize(obj: "StorableObject", protocol: SerializationProtocol) -> bin:
        pass

    @staticmethod
    def deserialize(binary_data: bin, protocol: SerializationProtocol) -> any:
        if protocol is SerializationProtocol.ProtoBuffers:
            pass

        if protocol is SerializationProtocol.FlatBuffers:
            pass

        if protocol is SerializationProtocol.Custom:
            pass

    @staticmethod
    def get_protobuf_schema():
        pass
