# third party
from torch import device

# relative
from ...core.common.serde import _deserialize
from ...core.common.serde import _serialize
from ...core.common.serde import recursive_serde_register

recursive_serde_register(
    device,
    serialize=lambda device_obj: _serialize(
        (device_obj.type, device_obj.index), to_bytes=True
    ),
    deserialize=lambda device_bytes: device(
        *_deserialize(device_bytes, from_bytes=True)
    ),
)
