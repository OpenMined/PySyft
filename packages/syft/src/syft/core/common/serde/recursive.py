# stdlib
import sys
from typing import Any

# syft absolute
import syft as sy

# relative
from ....proto.core.common.recursive_serde_pb2 import (
    RecursiveSerde as RecursiveSerde_PB,
)
from ....util import get_fully_qualified_name

TYPE_BANK = {}


def recursive_serde(cls: str) -> bool:
    if cls is None:
        return True
    return cls.__name__ in TYPE_BANK


def recursive_serde_register(
    cls: type,
    serialize=None,
    deserialize=None,
) -> None:
    if serialize is not None and deserialize is not None:
        nonrecursive = True
    else:
        nonrecursive = False

    serialize = serialize if nonrecursive else rs_object2proto
    deserialize = deserialize if nonrecursive else rs_proto2object

    attribute_list = getattr(cls, "__attr_allowlist__", None)
    serde_overrides = getattr(cls, "__serde_overrides__", {})

    TYPE_BANK[cls.__name__] = (
        nonrecursive,
        serialize,
        deserialize,
        attribute_list,
        serde_overrides,
    )


def rs_object2proto(self: Any) -> RecursiveSerde_PB:
    # if __attr_allowlist__ then only include attrs from that list
    msg = RecursiveSerde_PB(fully_qualified_name=get_fully_qualified_name(self))
    type_name = type(self).__name__

    nonrecursive, serialize, deserialize, attribute_list, serde_overrides = TYPE_BANK[
        type_name
    ]

    if nonrecursive:
        msg.nonrecursive_blob = serialize(self)
        return msg

    if attribute_list is None:
        attribute_list = self.__dict__.keys()

    for attr_name in attribute_list:
        if not hasattr(self, attr_name):
            raise ValueError(
                f"{attr_name} on {type(self)} does not exist, serialization aborted!"
            )

        field_obj = getattr(self, attr_name)
        transforms = serde_overrides.get(attr_name, None)

        if transforms is not None:
            field_obj = transforms[0](field_obj)

        serialized = sy.serialize(field_obj, to_bytes=True)

        msg.fields_name.append(attr_name)
        msg.fields_data.append(serialized)

    return msg


def rs_bytes2object(blob: bytes) -> Any:
    rs_message = RecursiveSerde_PB()
    rs_message.ParseFromString(blob)
    return rs_proto2object(rs_message)


def rs_proto2object(proto: RecursiveSerde_PB) -> Any:
    module_parts = proto.fully_qualified_name.split(".")
    klass = module_parts.pop()
    # clean this mess, Tudor
    if klass == "NoneType":
        class_type = type(None)
    else:
        class_type = getattr(sys.modules[".".join(module_parts)], klass)

    nonrecursive, serialize, deserialize, attribute_list, serde_overrides = TYPE_BANK[
        class_type.__name__
    ]

    if nonrecursive:
        return deserialize(proto.nonrecursive_blob)

    obj = class_type.__new__(class_type)  # type: ignore
    for attr_name, attr_bytes in zip(proto.fields_name, proto.fields_data):

        attr_value = sy.deserialize(attr_bytes, from_bytes=True)
        transforms = serde_overrides.get(attr_name, None)
        if transforms is None:
            setattr(obj, attr_name, attr_value)
        else:
            setattr(obj, attr_name, transforms[1](attr_value))

    return obj
