# stdlib
from collections.abc import Sequence
import os
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List

# third party
import capnp

CAPNP_START_MAGIC_HEADER = "capnp:"
CAPNP_END_MAGIC_HEADER = ":capnp"
CAPNP_START_MAGIC_HEADER_BYTES = CAPNP_START_MAGIC_HEADER.encode("utf-8")
CAPNP_END_MAGIC_HEADER_BYTES = CAPNP_END_MAGIC_HEADER.encode("utf-8")
CAPNP_REGISTRY: Dict[str, Callable] = {}

CapnpModule = capnp.lib.capnp._StructModule


def get_capnp_schema(schema_file: str) -> type:
    here = os.path.dirname(__file__)
    root_dir = Path(here) / ".." / ".." / ".." / "capnp"
    capnp_path = os.path.abspath(root_dir / schema_file)
    return capnp.load(str(capnp_path))


def chunk_bytes(
    data: Sequence, field_name: str, builder: capnp.lib.capnp._DynamicStructBuilder
) -> None:
    CHUNK_SIZE = int(5.12e8)  # capnp max for a List(Data) field
    list_size = len(data) // CHUNK_SIZE + 1
    data_lst = builder.init(field_name, list_size)
    END_INDEX = CHUNK_SIZE
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


def serde_magic_header(cls: type) -> bytes:
    return (
        CAPNP_START_MAGIC_HEADER_BYTES
        + cls.__name__.encode("utf-8")
        + CAPNP_END_MAGIC_HEADER_BYTES
    )
