# stdlib
from enum import EnumMeta
from typing import Any
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...proto.util.enum_type_pb2 import EnumType as EnumType_PB
from ...util import aggressive_set_attr


def GenerateEnumTypeWrapper(
    enum_type: EnumMeta, import_path: str
) -> None:
    class EnumTypeWrapper(StorableObject):
        def __init__(self, value: object):
            # set empty defaults, then this object will be used in construct_new_object
            # and the real id, tags and description can be stored on the wrapper
            super().__init__(
                data=value,
                id=UID(),
                tags=[],
                description="",
            )
            self.value = value

        def _data_object2proto(self) -> Any:
            proto = EnumType_PB()
            proto.member_name = self.value.name
            return proto

        @staticmethod
        def _data_proto2object(proto: Any) -> "EnumTypeWrapper":
            name = proto.member_name
            enum_obj = enum_type.__members__[name]
            return EnumTypeWrapper(value=enum_obj)

        @staticmethod
        def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
            return EnumType_PB

        @staticmethod
        def construct_new_object(
            id: UID,
            data: StorableObject,
            description: Optional[str],
            tags: Optional[List[str]],
        ) -> StorableObject:
            setattr(data, "_id", id)
            data.tags = tags
            data.description = description
            return data

    module_parts = import_path.split(".")
    klass = module_parts.pop()
    module_path = ".".join(module_parts)
    EnumTypeWrapper.__name__ = f"{klass}EnumTypeWrapper"
    EnumTypeWrapper.__module__ = module_path

    aggressive_set_attr(
        obj=enum_type, name="serializable_wrapper_type", attr=EnumTypeWrapper
    )