# stdlib
from typing import Any
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...util import aggressive_set_attr


# this will overwrite the .serializable_wrapper_type with an auto generated
# wrapper which will basically just hold the proto in a StorableObject
# and pass it through any time a protobuf is required
def GenerateProtobufWrapper(
    cls_pb: GeneratedProtocolMessageType, import_path: str
) -> None:
    class ProtobufWrapper(StorableObject):
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
            return self.data

        @staticmethod
        def _data_proto2object(proto: Any) -> "ProtobufWrapper":
            return ProtobufWrapper(value=proto)

        @staticmethod
        def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
            return cls_pb

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
    ProtobufWrapper.__name__ = f"{klass}ProtobufWrapper"
    ProtobufWrapper.__module__ = module_path

    aggressive_set_attr(
        obj=cls_pb, name="serializable_wrapper_type", attr=ProtobufWrapper
    )
