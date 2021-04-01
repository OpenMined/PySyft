"""This source file is used for production purposes."""
from app import create_app
import os

args = {
    "port": os.environ.get("GRID_NODE_PORT", 5000),
    "host": os.environ.get("GRID_NODE_HOST", "0.0.0.0"),
    "name": os.environ.get("GRID_NODE_NAME", "OpenMined"),
}
args_obj = type("args", (object,), args)()

app = create_app(args=args_obj)
