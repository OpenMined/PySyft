# stdlib
import os
from pathlib import Path

# third party
import capnp

# relative
from ..util._std_stream_capture import std_stream_capture


def get_capnp_schema(schema_file: str) -> type:
    here = os.path.dirname(__file__)
    root_dir = Path(here) / ".." / "capnp"
    capnp_path = os.path.abspath(root_dir / schema_file)

    with std_stream_capture():
        return capnp.load(str(capnp_path))
