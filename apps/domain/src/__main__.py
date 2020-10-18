"""
Note:

This file should be used only for development purposes.
Use the Flask built-in web server isn't suitable for production.
For production, we need to put it behind real web server able to communicate
with Flask through a WSGI protocol.
A common choice for that is Gunicorn.
"""

from app import create_app
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import argparse
import os
import json

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
    "--network",
    type=str,
    help="Grid Network address, e.g. --network=0.0.0.0:7000. Default is os.environ.get('NETWORK',None).",
    default=os.environ.get("NETWORK", None),
)

parser.add_argument(
    "--start_local_db",
    dest="start_local_db",
    action="store_true",
    help="If this flag is used a SQLAlchemy DB URI is generated to use a local db.",
)

parser.add_argument(
    "--id",
    type=str,
    help="PyGrid Node ID.",
    default=os.environ.get("NODE_ID", None),
)

parser.set_defaults(use_test_config=False)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.start_local_db:
        db_path = "sqlite:///databasenetwork.db"
        app = create_app(
            test_config={"SQLALCHEMY_DATABASE_URI": db_path},
        )
    else:
        app = create_app()

    _network = args.network
    _address = "http://{}:{}".format(args.host, args.port)

    server = pywsgi.WSGIServer(
        (args.host, args.port), app, handler_class=WebSocketHandler
    )
    server.serve_forever()
else:
    node_id = os.environ.get("NODE_ID", None)
    num_replicas = os.environ.get("N_REPLICAS", None)
    _address = os.environ.get("ADDRESS", None)
    _network = os.environ.get("NETWORK", None)

    app = create_app()
