# stdlib
import sys
from typing import Any
from typing import Callable
from typing import Optional
from typing import Type
from typing import Union

# syft absolute
import syft as sy

# relative
from ....proto.core.common.recursive_serde_pb2 import (
    RecursiveSerde as RecursiveSerde_PB,
)
from ....util import get_fully_qualified_name
from ....util import index_syft_by_module_name

TYPE_BANK = {}


def recursive_serde_register(
    cls: Union[object, type],
    serialize: Optional[Callable] = None,
    deserialize: Optional[Callable] = None,
) -> None:
    if not isinstance(cls, type):
        cls = type(cls)

    if serialize is not None and deserialize is not None:
        nonrecursive = True
    else:
        nonrecursive = False

    _serialize = serialize if nonrecursive else rs_object2proto
    _deserialize = deserialize if nonrecursive else rs_proto2object

    attribute_list = getattr(cls, "__attr_allowlist__", None)
    serde_overrides = getattr(cls, "__serde_overrides__", {})

    # without fqn duplicate class names overwrite
    fqn = f"{cls.__module__}.{cls.__name__}"
    TYPE_BANK[fqn] = (
        nonrecursive,
        _serialize,
        _deserialize,
        attribute_list,
        serde_overrides,
    )


def rs_object2proto(self: Any) -> RecursiveSerde_PB:
    # if __attr_allowlist__ then only include attrs from that list
    fqn = get_fully_qualified_name(self)
    msg = RecursiveSerde_PB(fully_qualified_name=fqn)
    if fqn not in TYPE_BANK:
        raise Exception(f"{fqn} not in TYPE_BANK")
    nonrecursive, serialize, deserialize, attribute_list, serde_overrides = TYPE_BANK[
        fqn
    ]

    if nonrecursive:
        if serialize is None:
            raise Exception(
                f"Cant serialize {type(self)} nonrecursive without serialize."
            )
        msg.nonrecursive_blob = serialize(self)
        return msg

    if attribute_list is None:
        attribute_list = self.__dict__.keys()

    for attr_name in sorted(attribute_list):
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
    # relative
    from .deserialize import _deserialize

    # clean this mess, Tudor
    module_parts = proto.fully_qualified_name.split(".")
    klass = module_parts.pop()

    class_type: Type = type(None)
    if klass != "NoneType":
        try:
            class_type = index_syft_by_module_name(proto.fully_qualified_name)  # type: ignore
        except Exception:  # nosec
            class_type = getattr(sys.modules[".".join(module_parts)], klass)

    if proto.fully_qualified_name not in TYPE_BANK:
        raise Exception(f"{proto.fully_qualified_name} not in TYPE_BANK")

    nonrecursive, serialize, deserialize, attribute_list, serde_overrides = TYPE_BANK[
        proto.fully_qualified_name
    ]

    if nonrecursive:
        if deserialize is None:
            raise Exception(
                f"Cant serialize {type(proto)} nonrecursive without serialize."
            )
        return deserialize(proto.nonrecursive_blob)

    obj = class_type.__new__(class_type)  # type: ignore
    for attr_name, attr_bytes in zip(proto.fields_name, proto.fields_data):
        attr_value = _deserialize(attr_bytes, from_bytes=True)
        transforms = serde_overrides.get(attr_name, None)
        if transforms is None:
            setattr(obj, attr_name, attr_value)
        else:
            setattr(obj, attr_name, transforms[1](attr_value))

    return obj
