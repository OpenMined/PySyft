# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ...core.common import UID
from ...core.common.serde.serializable import Serializable
from ...core.common.serde.serializable import bind_protobuf
from ...core.store.storeable_object import StorableObject
from ...util import aggressive_set_attr


# this will overwrite the .serializable_wrapper_type with an auto generated
# wrapper which will basically just hold the object whose type is C type class
def GenerateWrapper(
    ctype: type,
    import_path: str,
    protobuf_scheme: GeneratedProtocolMessageType,
    ctype_object2proto: CallableT,
    ctype_proto2object: CallableT,
    module_globals: dict
) -> None:
    @bind_protobuf
    class Wrapper(Serializable):
        def __init__(self, value: object):
            self.obj = value

        def _object2proto(self) -> Any:
            return ctype_object2proto(self.obj)

        @staticmethod
        def _proto2object(proto: Any) -> Any:
            obj = ctype_proto2object(proto)
            return Wrapper(value=obj)

        @staticmethod
        def get_protobuf_schema() -> GeneratedProtocolMessageType:
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
            return self.obj

    module_parts = import_path.split(".")
    klass = module_parts.pop()
    Wrapper.__name__ = f"{klass}Wrapper"
    Wrapper.__module__ = module_globals['__name__']
    module_globals[Wrapper.__name__] = Wrapper

    aggressive_set_attr(obj=ctype, name="serializable_wrapper_type", attr=Wrapper)