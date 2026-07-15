from .base import ProtocolCodec
from .v0 import ProtocolCodecV0
from .v1 import ProtocolCodecV1

# Ordered oldest -> newest. The last codec is the one the current dataset
# protocol uses (CODECS[-1].version == DATASET_PROTOCOL_VERSION); adding a new
# on-disk format means appending a ProtocolCodecV<n> here and bumping
# DATASET_PROTOCOL_VERSION.
CODECS = [ProtocolCodecV0, ProtocolCodecV1]

__all__ = [
    "ProtocolCodec",
    "ProtocolCodecV0",
    "ProtocolCodecV1",
    "CODECS",
]
