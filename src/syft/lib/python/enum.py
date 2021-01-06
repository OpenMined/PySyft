# stdlib
import importlib
from typing import Any
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...proto.util.enum_pb2 import EnumMember as EnumMember_PB
from ...util import aggressive_set_attr


# this will overwrite the .serializable_wrapper_type with an auto generated
# wrapper which will basically just hold the object whose type is C type class
def GenerateEnumLikeWrapper(enum_like_type: type, import_path: str) -> None:
    class EnumLikeWrapper(StorableObject):
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
            proto = EnumMember_PB()
            proto.member_name = str(self.value)
            return proto

        @staticmethod
        def _data_proto2object(proto: Any) -> "EnumLikeWrapper":
            name = proto.member_name.split(".")
            module = importlib.import_module(".".join(name[:-1]))
            obj = getattr(module, name[-1])
            return EnumLikeWrapper(value=obj)

        @staticmethod
        def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
            return EnumMember_PB

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

        def upcast(self) -> Any:
            return self.value

    module_parts = import_path.split(".")
    klass = module_parts.pop()
    module_path = ".".join(module_parts)
    EnumLikeWrapper.__name__ = f"{klass}EnumLikeWrapper"
    EnumLikeWrapper.__module__ = module_path

    aggressive_set_attr(
        obj=enum_like_type, name="serializable_wrapper_type", attr=EnumLikeWrapper
    )
