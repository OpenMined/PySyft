# stdlib
from typing import Any
from typing import Callable as CallableT

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft

# syft relative
from .core.common.serde.serializable import Serializable
from .core.common.serde.serializable import bind_protobuf
from .util import aggressive_set_attr

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
    @bind_protobuf
    class Wrapper(Serializable):
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

    # set __module__ and __name__
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
