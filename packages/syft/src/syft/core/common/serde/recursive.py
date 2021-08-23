# stdlib
from typing import Any
from typing import Dict as DictType
from typing import List

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
from syft import deserialize
from syft import serialize

# relative
from ....core.common.serde.serializable import bind_protobuf
from ....lib.python import Dict
from ....proto.core.common.recursive_serde_pb2 import (
    RecursiveSerde as RecursiveSerde_PB,
)
from ....util import get_fully_qualified_name
from ....util import index_syft_by_module_name
from .serializable import Serializable


@bind_protobuf
class RecursiveSerde(Serializable):
    """If you subclass from this object and put that subclass in the syft classpath somehow, then
    you'll be able to serialize it without having to create a custom protobuf. Be careful with this
    though, because it's going to include all attributes by default (including private data if
    it's there)."""

    # put attr names here - set this to None to include all attrs (not recommended)
    __attr_allowlist__: List[str] = []
    __serde_overrides__: DictType[Any, Any] = {}

    def _object2proto(self) -> RecursiveSerde_PB:

        # if __attr_allowlist__ then only include attrs from that list
        if self.__attr_allowlist__ is not None:
            attrs = {}
            for attr_name in self.__attr_allowlist__:
                if hasattr(self, attr_name):
                    if self.__serde_overrides__.get(attr_name, None) is None:
                        attrs[attr_name] = getattr(self, attr_name)
                    else:
                        attrs[attr_name] = self.__serde_overrides__[attr_name][0](
                            getattr(self, attr_name)
                        )
        # else include all attrs
        else:
            attrs = self.__dict__  # type: ignore

        return RecursiveSerde_PB(
            data=serialize(Dict(attrs), to_bytes=True),
            fully_qualified_name=get_fully_qualified_name(self),
        )

    @staticmethod
    def _proto2object(proto: RecursiveSerde_PB) -> "RecursiveSerde":

        attrs = dict(deserialize(proto.data, from_bytes=True))

        class_type = index_syft_by_module_name(proto.fully_qualified_name)

        obj = class_type.__new__(class_type)  # type: ignore

        for attr_name, attr_value in attrs.items():
            if obj.__serde_overrides__.get(attr_name, None) is None:
                setattr(obj, attr_name, attr_value)
            else:
                setattr(
                    obj, attr_name, obj.__serde_overrides__[attr_name][1](attr_value)
                )

        return obj

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RecursiveSerde_PB
