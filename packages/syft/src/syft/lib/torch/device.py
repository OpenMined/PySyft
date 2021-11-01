# stdlib
from typing import Any
from typing import Optional

# third party
from torch import device

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.torch.device_pb2 import Device as Device_PB

# use -2 to represent index=None
INDEX_NONE = -2


def object2proto(obj: device) -> "Device_PB":
    proto = Device_PB()
    proto.type = obj.type
    proto.index = INDEX_NONE if obj.index is None else obj.index
    return proto


def proto2object(proto: "Device_PB") -> Any:
    device_type = proto.type
    index: Optional[int] = None if proto.index == INDEX_NONE else proto.index
    obj = device(device_type, index)
    return obj


serializable(generate_wrapper=True)(
    wrapped_type=device,
    import_path="torch.device",
    protobuf_scheme=Device_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
