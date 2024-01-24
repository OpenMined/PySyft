# stdlib
import os
from pathlib import Path

# third party
import capnp


def get_capnp_schema(schema_file: str) -> type:
    here = os.path.dirname(__file__)
    root_dir = Path(here) / ".." / "capnp"
    capnp_path = os.path.abspath(root_dir / schema_file)
    return capnp.load(str(capnp_path))
