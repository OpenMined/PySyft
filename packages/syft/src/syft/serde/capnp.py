# stdlib
from importlib.resources import as_file
from importlib.resources import files

# third party
import capnp

# relative
from ..util._std_stream_capture import std_stream_capture


def get_capnp_schema(schema_file: str) -> type:
    with as_file(files("syft.capnp").joinpath(schema_file)) as capnp_path:
        with std_stream_capture():
            return capnp.load(str(capnp_path.absolute()))
