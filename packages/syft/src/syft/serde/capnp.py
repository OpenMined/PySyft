# stdlib
import os
from pathlib import Path

# third party
import capnp
from hide_warnings import hide_warnings


@hide_warnings
def load_capnp(capnp_path):

    # # suppress warnings for a moment, specifically the common Jupyter warning:
    # # kj/filesystem-disk-unix.c++:1734: warning: PWD environment variable
    # # doesn't match current directory; pwd = /big/local/path/on/your/machine
    # warnings.filterwarnings("ignore")

    return capnp.load(str(capnp_path))


def get_capnp_schema(schema_file: str) -> type:
    here = os.path.dirname(__file__)
    root_dir = Path(here) / ".." / "capnp"
    capnp_path = os.path.abspath(root_dir / schema_file)
    return load_capnp(capnp_path)
