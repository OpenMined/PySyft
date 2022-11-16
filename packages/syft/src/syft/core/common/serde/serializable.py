# stdlib
from typing import Any
from typing import Callable as CallableT
import warnings

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ....util import aggressive_set_attr
from ....util import get_loaded_syft
from .capnp import CAPNP_REGISTRY


# this will overwrite the ._sy_serializable_wrapper_type with an auto generated
# wrapper which will basically just hold the object being wrapped.
def GenerateWrapper(
    wrapped_type: type,
    import_path: str,
    protobuf_scheme: GeneratedProtocolMessageType,
    type_object2proto: CallableT,
    type_proto2object: CallableT,
) -> None:
    @serializable()
    class Wrapper:
        def __init__(self, value: object):
            self.obj = value

        def _object2proto(self) -> Any:
            return type_object2proto(self.obj)

        @staticmethod
        def _proto2object(proto: Any) -> Any:
            return type_proto2object(proto)

        @staticmethod
        def get_protobuf_schema() -> GeneratedProtocolMessageType:
            return protobuf_scheme

        def upcast(self) -> Any:
            return self.obj

        @staticmethod
        def wrapped_type() -> type:
            return wrapped_type

    # TODO: refactor like proxy class to get correct name
    # WARNING: Changing this can break the Wrapper lookup during deserialize
    module_parts = import_path.split(".")
    klass = module_parts.pop()
    Wrapper.__name__ = f"{klass}Wrapper"
    Wrapper.__module__ = f"syft.wrappers.{'.'.join(module_parts)}"

    syft_module = get_loaded_syft()
    module_type = type(syft_module)
    # create a fake module `wrappers` under `syft`
    if "wrappers" not in syft_module.__dict__:
        syft_module.__dict__["wrappers"] = module_type(name="wrappers")
    # for each part of the path, create a fake module and add it to it's parent
    parent = syft_module.__dict__["wrappers"]
    for n in module_parts:
        if n not in parent.__dict__:
            parent.__dict__[n] = module_type(name=n)
        parent = parent.__dict__[n]
    # finally add our wrapper class to the end of the path
    parent.__dict__[Wrapper.__name__] = Wrapper

    aggressive_set_attr(
        obj=wrapped_type, name="_sy_serializable_wrapper_type", attr=Wrapper
    )


def GenerateProtobufWrapper(
    cls_pb: GeneratedProtocolMessageType, import_path: str
) -> None:
    def object2proto(obj: Any) -> Any:
        return obj

    def proto2object(proto: Any) -> Any:
        return proto

    GenerateWrapper(
        wrapped_type=cls_pb,
        import_path=import_path,
        protobuf_scheme=cls_pb,
        type_object2proto=object2proto,
        type_proto2object=proto2object,
    )


def serializable(
    generate_wrapper: bool = False,
    protobuf_object: bool = False,
    recursive_serde: bool = False,
    capnp_bytes: bool = False,
) -> Any:
    def rs_decorator(cls: Any) -> Any:
        # relative
        from .recursive import rs_get_protobuf_schema
        from .recursive import rs_object2proto
        from .recursive import rs_proto2object

        if not hasattr(cls, "__attr_allowlist__"):
            warnings.warn(
                f"__attr_allowlist__ not defined for type {cls.__name__},"
                " even if it uses recursive serde, defaulting on the empty list."
            )
            cls.__attr_allowlist__ = []

        if not hasattr(cls, "__serde_overrides__"):
            cls.__serde_overrides__ = {}

        cls._object2proto = rs_object2proto
        cls._proto2object = staticmethod(rs_proto2object)
        cls.get_protobuf_schema = staticmethod(rs_get_protobuf_schema)
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

    if generate_wrapper:
        return GenerateWrapper

    if recursive_serde:
        return rs_decorator

    return serializable_decorator
