from typing import TYPE_CHECKING

from .base import ProtocolCodec
from .v0 import DatasetConfigV0, ProtocolCodecV0
from .v1 import DatasetConfigV1, ProtocolCodecV1

if TYPE_CHECKING:
    from ..config import SyftBoxConfig

# Ordered oldest -> newest. The last codec is the one the current dataset
# protocol uses (CODECS[-1].version == DATASET_PROTOCOL_VERSION); adding a new
# on-disk format means appending a ProtocolCodecV<n> here and bumping
# DATASET_PROTOCOL_VERSION.
CODECS = [ProtocolCodecV0, ProtocolCodecV1]


def dataset_config_for_protocol(
    protocol_version: str, syftbox_config: "SyftBoxConfig"
) -> "DatasetConfigV0 | DatasetConfigV1":
    """The layout that stores a given protocol version (mirrors codec selection)."""
    for codec_cls in CODECS:
        config_cls = codec_cls.dataset_config_cls
        if protocol_version in config_cls.protocol_versions:
            return config_cls(syftbox_config)
    raise ValueError(f"No dataset layout for protocol version {protocol_version!r}")


__all__ = [
    "DatasetConfigV0",
    "DatasetConfigV1",
    "ProtocolCodec",
    "ProtocolCodecV0",
    "ProtocolCodecV1",
    "CODECS",
    "dataset_config_for_protocol",
]
