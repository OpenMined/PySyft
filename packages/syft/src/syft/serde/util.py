# stdlib
from sys import platform

# third party
from capnp.lib.capnp import _DynamicStructBuilder


def get_size(thing: _DynamicStructBuilder | int) -> int:
    if isinstance(thing, int):
        return thing
    return thing.total_size.word_count


def compatible_with_large_file_writes_capnp(thing: _DynamicStructBuilder | int) -> bool:
    if platform in ["darwin", "win32"]:
        return False
    else:
        return get_size(thing) > 50000000  # roughly 0.5GB
