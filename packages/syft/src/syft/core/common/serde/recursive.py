# stdlib
import sys
from typing import Any
from typing import Callable
from typing import Optional
from typing import Type
from typing import Union

# third party
from capnp.lib.capnp import _DynamicStructBuilder

# syft absolute
import syft as sy

# relative
from ....util import get_fully_qualified_name
from ....util import index_syft_by_module_name
from .capnp import get_capnp_schema

TYPE_BANK = {}

recursive_scheme = get_capnp_schema("recursive_serde.capnp").RecursiveSerde  # type: ignore


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


def rs_object2proto(self: Any) -> _DynamicStructBuilder:
    msg = recursive_scheme.new_message()
    fqn = get_fully_qualified_name(self)

    if fqn not in TYPE_BANK:
        raise Exception(f"{fqn} not in TYPE_BANK")

    msg.fullyQualifiedName = fqn
    nonrecursive, serialize, deserialize, attribute_list, serde_overrides = TYPE_BANK[
        fqn
    ]

    if nonrecursive:
        if serialize is None:
            raise Exception(
                f"Cant serialize {type(self)} nonrecursive without serialize."
            )
        msg.nonrecursiveBlob = serialize(self)
        return msg

    if attribute_list is None:
        attribute_list = self.__dict__.keys()

    msg.init("fieldsName", len(attribute_list))
    msg.init("fieldsData", len(attribute_list))

    for idx, attr_name in enumerate(sorted(attribute_list)):
        if not hasattr(self, attr_name):
            raise ValueError(
                f"{attr_name} on {type(self)} does not exist, serialization aborted!"
            )

        field_obj = getattr(self, attr_name)
        transforms = serde_overrides.get(attr_name, None)

        if transforms is not None:
            field_obj = transforms[0](field_obj)

        serialized = sy.serialize(field_obj, to_bytes=True)

        msg.fieldsName[idx] = attr_name
        msg.fieldsData[idx] = serialized

    return msg


def rs_bytes2object(blob: bytes) -> Any:
    MAX_TRAVERSAL_LIMIT = 2**64 - 1

    with recursive_scheme.from_bytes(  # type: ignore
        blob, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
    ) as msg:
        return rs_proto2object(msg)


def rs_proto2object(proto: _DynamicStructBuilder) -> Any:
    # relative
    from .deserialize import _deserialize

    # clean this mess, Tudor
    module_parts = proto.fullyQualifiedName.split(".")
    klass = module_parts.pop()

    class_type: Type = type(None)
    if klass != "NoneType":
        try:
            class_type = index_syft_by_module_name(proto.fullyQualifiedName)  # type: ignore
        except Exception:  # nosec
            class_type = getattr(sys.modules[".".join(module_parts)], klass)

    if proto.fullyQualifiedName not in TYPE_BANK:
        raise Exception(f"{proto.fully_qualified_name} not in TYPE_BANK")

    nonrecursive, serialize, deserialize, attribute_list, serde_overrides = TYPE_BANK[
        proto.fullyQualifiedName
    ]

    if nonrecursive:
        if deserialize is None:
            raise Exception(
                f"Cant serialize {type(proto)} nonrecursive without serialize."
            )
        return deserialize(proto.nonrecursiveBlob)

    kwargs = {}

    for attr_name, attr_bytes in zip(proto.fieldsName, proto.fieldsData):
        attr_value = _deserialize(attr_bytes, from_bytes=True)
        transforms = serde_overrides.get(attr_name, None)

        if transforms is not None:
            attr_value = transforms[1](attr_value)
        kwargs[attr_name] = attr_value

    if hasattr(class_type, "serde_constructor"):
        return getattr(class_type, "serde_constructor")(kwargs)

    obj = class_type.__new__(class_type)  # type: ignore
    for attr_name, attr_value in kwargs.items():
        setattr(obj, attr_name, attr_value)

    return obj
