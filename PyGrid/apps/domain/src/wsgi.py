"""This source file is used for production purposes."""
# stdlib
import os

# third party
from app import create_app

args = {
    "port": os.environ.get("GRID_NODE_PORT", 5000),
    "host": os.environ.get("GRID_NODE_HOST", "0.0.0.0"),
    "name": os.environ.get("GRID_NODE_NAME", "OpenMined"),
    "start_local_db": os.environ.get("LOCAL_DATABASE", False),
}
args_obj = type("args", (object,), args)()

app = create_app(args=args_obj)
