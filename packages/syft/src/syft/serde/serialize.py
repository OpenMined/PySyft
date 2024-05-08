# stdlib
import tempfile
from typing import Any

# relative
from .util import compatible_with_large_file_writes_capnp


def _serialize(
    obj: object,
    to_proto: bool = True,
    to_bytes: bool = False,
    for_hashing: bool = False,
) -> Any:
    # relative
    from .recursive import rs_object2proto

    proto = rs_object2proto(obj, for_hashing=for_hashing)
    if to_bytes:
        if compatible_with_large_file_writes_capnp(proto):
            with tempfile.TemporaryFile() as tmp_file:
                # Write data to a file to save RAM
                proto.write(tmp_file)
                # proto in memory, and bytes in file
                del proto
                # bytes in file
                tmp_file.seek(0)
                return tmp_file.read()
        else:
            res = proto.to_bytes()
            return res

    if to_proto:
        return proto
