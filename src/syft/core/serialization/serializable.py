from typing import Optional, Dict, Callable
from . import SerializationProtocol, SerializationStore

DEFAULT_PROTOCOL = SerializationProtocol.ProtoBuffers

def get_


class Serializable:
    @staticmethod
    def serialize(
        obj: "Serializable", protocol: Optional[int] = DEFAULT_PROTOCOL
    ) -> bin:
        obj_type = type(obj)
        schema = SerializationStore.type_to_schema[obj_type][protocol](obj)
        return schema.tobinary()

    @staticmethod
    def deserialize(
        data: any, protocol: Optional[int] = DEFAULT_PROTOCOL
    ) -> "Serializable":
        schema_type = type(data)
        obj = SerializationStore.schema_to_type[protocol][schema_type](data)
        return obj

    @staticmethod
    def to_schema(obj: "Serializable") -> Dict[Callable]:
        raise NotImplementedError

    @staticmethod
    def from_schema(schema: any) -> Dict[Callable]:
        raise NotImplementedError

    @staticmethod
    def get_wrapper_type(self):
        raise NotImplementedError
