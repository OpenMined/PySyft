# stdlib
from typing import Any

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from .ctype import GenerateWrapper


def GenerateProtobufWrapper(
    cls_pb: "GeneratedProtocolMessageType", import_path: str
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
