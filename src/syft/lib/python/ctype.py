# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...util import aggressive_set_attr


# this will overwrite the .serializable_wrapper_type with an auto generated
# wrapper which will basically just hold the object whose type is C type class
def GenerateCTypeWrapper(
    ctype: type,
    import_path: str,
    protobuf_scheme: GeneratedProtocolMessageType,
    ctype_object2proto: CallableT,
    ctype_proto2object: CallableT,
) -> None:
    class CTypeWrapper(StorableObject):
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
            obj = self.value
            return ctype_object2proto(obj)

        @staticmethod
        def _data_proto2object(proto: Any) -> Any:
            obj = ctype_proto2object(proto)
            return CTypeWrapper(value=obj)

        @staticmethod
        def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
            return protobuf_scheme

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
    CTypeWrapper.__name__ = f"{klass}CTypeWrapper"
    CTypeWrapper.__module__ = module_path

    aggressive_set_attr(obj=ctype, name="serializable_wrapper_type", attr=CTypeWrapper)
