# stdlib
from typing import Any
from typing import Callable as CallableT

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft

# relative
from ....util import aggressive_set_attr
from .capnp import CAPNP_REGISTRY
from .recursive import recursive_serde_register

module_type = type(syft)


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
    # create a fake module `wrappers` under `syft`
    if "wrappers" not in syft.__dict__:
        syft.__dict__["wrappers"] = module_type(name="wrappers")
    # for each part of the path, create a fake module and add it to it's parent
    parent = syft.__dict__["wrappers"]
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

    if generate_wrapper:
        return GenerateWrapper

    if recursive_serde:
        return rs_decorator

    return serializable_decorator
