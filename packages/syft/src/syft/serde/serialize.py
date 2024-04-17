# stdlib
import tempfile
from typing import Any


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
        with tempfile.TemporaryFile() as tmp_file:
            # Write data to a file to save RAM
            proto.write(tmp_file)
            del proto
            tmp_file.seek(0)
            return tmp_file.read()
        # return proto.to_bytes()

    if to_proto:
        return proto
    
