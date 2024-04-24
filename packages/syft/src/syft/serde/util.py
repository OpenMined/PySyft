# stdlib
from sys import platform

# third party
from capnp.lib.capnp import _DynamicStructBuilder


def compatible_with_large_file_writes_capnp(msg: _DynamicStructBuilder) -> bool:
    if platform in ["darwin", "win32"]:
        return False
    else:
        return msg.total_size.word_count > 50000000  # roughly 0.5GB
