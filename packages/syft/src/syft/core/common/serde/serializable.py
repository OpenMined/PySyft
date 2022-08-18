# stdlib
from typing import Any

# syft absolute
import syft

# relative
from .capnp import CAPNP_REGISTRY
from .recursive import recursive_serde_register

module_type = type(syft)


def serializable(
    recursive_serde: bool = False,
    capnp_bytes: bool = False,
) -> Any:
    def rs_decorator(cls: Any) -> Any:
        recursive_serde_register(cls)
        return cls

    def serializable_decorator(cls: Any) -> Any:
        protobuf_schema = cls.get_protobuf_schema()
        # overloading a protobuf by adding multiple classes and we will check the
        # obj_type string later to dispatch to the correct one
        if hasattr(protobuf_schema, "schema2type"):
            if isinstance(protobuf_schema.schema2type, list):
                protobuf_schema.schema2type.append(cls)
            else:
                protobuf_schema.schema2type = [protobuf_schema.schema2type, cls]
        else:
            protobuf_schema.schema2type = cls
        return cls

    def capnp_decorator(cls: Any) -> Any:
        # register deserialize with the capnp registry
        CAPNP_REGISTRY[cls.__name__] = cls._bytes2object
        return cls

    if capnp_bytes:
        return capnp_decorator

    if recursive_serde:
        return rs_decorator

    return serializable_decorator
