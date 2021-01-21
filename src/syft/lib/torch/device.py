# stdlib
from typing import Any

# third party
from torch import device

# syft relative
from ...proto.lib.torch.device_pb2 import Device as Device_PB
from ..python.ctype import GenerateCTypeWrapper


def object2proto(obj: "device") -> "Device_PB":
    proto = Device_PB()
    proto.type = obj.type
    proto.index = obj.index
    return proto


def proto2object(proto: "Device_PB") -> Any:
    device_type = proto.type
    index = proto.index
    obj = device(device_type, index)
    return obj


GenerateCTypeWrapper(
    ctype=device,
    import_path="torch.device",
    protobuf_scheme=Device_PB,
    ctype_object2proto=object2proto,
    ctype_proto2object=proto2object,
)
