from .base import ProtocolCodec
from .v0 import ProtocolCodecV0

# Ordered oldest -> newest. The last codec is the one the current job protocol
# uses (CODECS[-1].version == JOB_PROTOCOL_VERSION); adding a new on-disk format
# means appending a ProtocolCodecV<n> here and bumping JOB_PROTOCOL_VERSION.
CODECS = [ProtocolCodecV0]

__all__ = [
    "ProtocolCodec",
    "ProtocolCodecV0",
    "CODECS",
]
