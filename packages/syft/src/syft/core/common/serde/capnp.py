# stdlib
import os
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

# third party
import capnp
from capnp.lib.capnp import _DynamicStructBuilder
import numpy as np

# relative
from .arrow import arrow_deserialize
from .arrow import arrow_serialize

CAPNP_START_MAGIC_HEADER = "capnp:"
CAPNP_END_MAGIC_HEADER = ":capnp"
CAPNP_START_MAGIC_HEADER_BYTES = CAPNP_START_MAGIC_HEADER.encode("utf-8")
CAPNP_END_MAGIC_HEADER_BYTES = CAPNP_END_MAGIC_HEADER.encode("utf-8")
PROTOBUF_START_MAGIC_HEADER = "protobuf:"
PROTOBUF_START_MAGIC_HEADER_BYTES = PROTOBUF_START_MAGIC_HEADER.encode("utf-8")
CAPNP_REGISTRY: Dict[str, Callable] = {}

CapnpModule = capnp.lib.capnp._StructModule


def create_protobuf_magic_header() -> str:
    return f"{PROTOBUF_START_MAGIC_HEADER}"


def get_capnp_schema(schema_file: str) -> type:
    here = os.path.dirname(__file__)
    root_dir = Path(here) / ".." / ".." / ".." / "capnp"
    capnp_path = os.path.abspath(root_dir / schema_file)
    return capnp.load(str(capnp_path))


def chunk_bytes(data: bytes, field_name: str, builder: _DynamicStructBuilder) -> None:
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


def capnp_deserialize(
    msg: Union[_DynamicStructBuilder, bytes], from_bytes: bool = False
) -> np.ndarray:
    array_msg: _DynamicStructBuilder
    if from_bytes:
        schema = get_capnp_schema(schema_file="array.capnp")
        array_struct: CapnpModule = schema.Array  # type: ignore
        # https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize
        MAX_TRAVERSAL_LIMIT = 2**64 - 1
        # to pack or not to pack?
        # array_msg = array_struct.from_bytes(buf, traversal_limit_in_words=2 ** 64 - 1)
        array_msg = array_struct.from_bytes_packed(  # type: ignore
            msg, traversal_limit_in_words=MAX_TRAVERSAL_LIMIT
        )
    else:
        array_msg = msg

    # TODO: remove arrayMetadata?
    # array_metadata = array_msg.arrayMetadata
    obj = arrow_deserialize(combine_bytes(array_msg.array))

    return obj


# TODO: move to sy.serialize interface, when protobuf for numpy is removed.
def capnp_serialize(obj: np.ndarray, to_bytes: bool = False) -> _DynamicStructBuilder:
    schema = get_capnp_schema(schema_file="array.capnp")
    array_struct: CapnpModule = schema.Array  # type: ignore
    array_msg = array_struct.new_message()  # type: ignore
    metadata_schema = array_struct.TensorMetadata  # type: ignore
    array_metadata = metadata_schema.new_message()

    obj_bytes = arrow_serialize(obj)
    chunk_bytes(obj_bytes, "array", array_msg)  # type: ignore
    array_metadata.dtype = str(obj.dtype)
    # array_metadata.decompressedSize = obj_decompressed_size

    array_msg.arrayMetadata = array_metadata

    if not to_bytes:
        return array_msg
    else:
        return array_msg.to_bytes_packed()


class CapnpMagicBytesNotFound(Exception):
    pass


def deserialize_capnp(buf: bytes) -> Any:
    # only search 1000 bytes to prevent wasting time on large files
    search_range = 1000
    header_bytes = buf[0:search_range]
    chars = bytearray()
    # filter header bytes
    for i in header_bytes:
        # only allow ascii letters or : in headers and class name to prevent lookup
        # breaking somehow, when packing weird stuff like \x03 ends up in the string
        # e.g. PhiTensor -> ND\x03imEntityPhiTensor
        if i in range(65, 91) or i in range(97, 123) or i == 58:
            chars.append(i)
    header_bytes = bytes(chars)

    proto_start_index = header_bytes.find(PROTOBUF_START_MAGIC_HEADER_BYTES)
    start_index = header_bytes.find(CAPNP_START_MAGIC_HEADER_BYTES)
    if proto_start_index != -1 and (proto_start_index < start_index):
        # we have protobuf on the outside
        raise CapnpMagicBytesNotFound(
            f"protobuf Magic Header {PROTOBUF_START_MAGIC_HEADER} found in bytes"
        )
    if start_index == -1:
        raise CapnpMagicBytesNotFound(
            f"capnp Magic Header {CAPNP_START_MAGIC_HEADER}" + "not found in bytes"
        )
    start_index += len(CAPNP_START_MAGIC_HEADER_BYTES)
    end_index = start_index + header_bytes[start_index:].index(
        CAPNP_END_MAGIC_HEADER_BYTES
    )
    class_name_bytes = header_bytes[start_index:end_index]
    class_name = class_name_bytes.decode("utf-8")

    if end_index <= start_index:
        raise ValueError("End Index should always be greater than Start index")

    if class_name not in CAPNP_REGISTRY:
        raise Exception(
            f"Found capnp Magic Header: {CAPNP_START_MAGIC_HEADER} "
            + f"and Class {class_name} but no mapping in capnp registry {CAPNP_REGISTRY}"
        )
    return CAPNP_REGISTRY[class_name](buf=buf)
