#!/bin/env python
"""
    Grid Node is a Socket/HTTP server used to manage / compute data remotely.
"""
import json
import os
import requests
import sys

import argparse

from app import create_app, socketio


parser = argparse.ArgumentParser(description="Run Grid Node application.")

parser.add_argument(
    "--id",
    type=str,
    help="Grid node ID, e.g. --id=alice. Default is os.environ.get('GRID_WS_ID', None).",
    default=os.environ.get("GRID_WS_ID", None),
)

parser.add_argument(
    "--port",
    "-p",
    type=int,
    help="Port number of the socket.io server, e.g. --port=8777. Default is os.environ.get('GRID_WS_PORT', None).",
    default=os.environ.get("GRID_WS_PORT", None),
)

parser.add_argument(
    "--host",
    type=str,
    help="Grid node host, e.g. --host=0.0.0.0. Default is os.environ.get('GRID_WS_HOST','0.0.0.0').",
    default=os.environ.get("GRID_WS_HOST", "0.0.0.0"),
)

parser.add_argument(
    "--gateway_url",
    type=str,
    help="Address used to join a Grid Network. This argument is optional. Default is os.environ.get('GRID_NETWORK_URL', None).",
    default=os.environ.get("GRID_NETWORK_URL", None),
)

parser.add_argument(
    "--use_test_config",
    dest="use_test_config",
    action="store_true",
    help="If this flag is used a SQLAlchemy DB URI is generated to use a local db.",
)

parser.set_defaults(use_test_config=False)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.use_test_config:
        db_path = "sqlite:///database{}.db".format(args.id)
        app = create_app(debug=False, test_config={"SQLALCHEMY_DATABASE_URI": db_path})
    else:
        app = create_app(debug=False)

    # If using a Gateway URL start the connection
    if args.gateway_url is not None:
        requests.post(
            os.path.join(args.gateway_url, "join"),
            data=json.dumps(
                {
                    "node-id": args.id,
                    "node-address": "http://{}:{}".format(args.host, args.port),
                }
            ),
        )

    socketio.run(app, host=args.host, port=args.port)
