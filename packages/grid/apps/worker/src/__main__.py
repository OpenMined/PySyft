"""
Note:

This file should be used only for development purposes.
Use the Flask built-in web server isn't suitable for production.
For production, we need to put it behind real web server able to communicate
with Flask through a WSGI protocol.
A common choice for that is Gunicorn.
"""

# stdlib
import argparse
import json
import os

# third party
from app import create_app
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

parser = argparse.ArgumentParser(description="Run PyGrid application.")


parser.add_argument(
    "--port",
    "-p",
    type=int,
    help="Port number of the socket server, e.g. --port=5000. Default is os.environ.get('GRID_NODE_PORT', 5000).",
    default=os.environ.get("GRID_NODE_PORT", 5000),
)

parser.add_argument(
    "--host",
    type=str,
    help="Grid node host, e.g. --host=0.0.0.0. Default is os.environ.get('GRID_NODE_HOST','0.0.0.0').",
    default=os.environ.get("GRID_NODE_HOST", "0.0.0.0"),
)


parser.add_argument(
    "--name",
    type=str,
    help="Grid node name, e.g. --name=OpenMined. Default is os.environ.get('GRID_NODE_NAME','OpenMined').",
    default=os.environ.get("GRID_NODE_NAME", "OpenMined"),
)

parser.add_argument(
    "--start_local_db",
    dest="start_local_db",
    action="store_true",
    help="If this flag is used a SQLAlchemy DB URI is generated to use a local db.",
)

parser.set_defaults(use_test_config=False)


if __name__ == "__main__":
    args = parser.parse_args()

    app = create_app(args)
    _address = "http://{}:{}".format(args.host, args.port)

    server = pywsgi.WSGIServer(
        (args.host, args.port), app, handler_class=WebSocketHandler
    )
    server.serve_forever()
