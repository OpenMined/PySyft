# stdlib
import os
from pathlib import Path
from typing import List

# third party
import capnp
from capnp.lib.capnp import _DynamicStructBuilder


def get_capnp_schema(schema_file: str) -> type:
    here = os.path.dirname(__file__)
    root_dir = Path(here) / ".." / ".." / ".." / "capnp"
    capnp_path = os.path.abspath(root_dir / schema_file)
    return capnp.load(str(capnp_path))


def chunk_bytes(data: bytes, field_name: str, builder: _DynamicStructBuilder) -> None:
    CHUNK_SIZE = int(5.12e8)  # capnp max for a List(Data) field
    list_size = len(data) // CHUNK_SIZE + 1
    data_lst = builder.init(field_name, list_size)
    for idx in range(list_size):
        START_INDEX = idx * CHUNK_SIZE
        END_INDEX = min(START_INDEX + CHUNK_SIZE, len(data))
        data_lst[idx] = data[START_INDEX:END_INDEX]


def combine_bytes(capnp_list: List[bytes]) -> bytes:
    # TODO: make sure this doesn't copy, perhaps allocate a fixed size buffer
    # and move the bytes into it as we go
    bytes_value = b""
    for value in capnp_list:
        bytes_value += value
    return bytes_value
