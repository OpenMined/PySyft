# third party
import torch

# relative
from ...core.common.serde import _deserialize
from ...core.common.serde import _serialize
from ...core.common.serde import recursive_serde_register

recursive_serde_register(
    torch.Size,
    serialize=lambda torch_size: _serialize(tuple(torch_size), to_bytes=True),
    deserialize=lambda torch_size_bytes: torch.Size(
        _deserialize(torch_size_bytes, from_bytes=True)
    ),
)
