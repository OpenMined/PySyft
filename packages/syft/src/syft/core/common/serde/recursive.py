# stdlib
from typing import Any

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
from syft import deserialize
from syft import serialize

# relative
from ....lib.python import Dict
from ....proto.core.common.recursive_serde_pb2 import (
    RecursiveSerde as RecursiveSerde_PB,
)
from ....util import get_fully_qualified_name
from ....util import index_syft_by_module_name


def rs_object2proto(self: Any) -> RecursiveSerde_PB:
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


def rs_proto2object(proto: RecursiveSerde_PB) -> Any:
    attrs = dict(deserialize(proto.data, from_bytes=True))

    class_type = index_syft_by_module_name(proto.fully_qualified_name)

    obj = class_type.__new__(class_type)  # type: ignore

    for attr_name, attr_value in attrs.items():
        if obj.__serde_overrides__.get(attr_name, None) is None:
            setattr(obj, attr_name, attr_value)
        else:
            setattr(obj, attr_name, obj.__serde_overrides__[attr_name][1](attr_value))

    return obj


def rs_get_protobuf_schema() -> GeneratedProtocolMessageType:
    return RecursiveSerde_PB
