#!/bin/env python
"""
    Grid Node is a Socket/HTTP server used to manage / compute data remotely.
"""
import json
import os
import requests
import sys

import argparse

from app import create_app

from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

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
    help="Grid node host, e.g. --host=0.0.0.0. Default is os.environ.get('GRID_WS_HOST','http://0.0.0.0').",
    default=os.environ.get("GRID_WS_HOST", "0.0.0.0"),
)

parser.add_argument(
    "--gateway_url",
    type=str,
    help="Address used to join a Grid Network. This argument is optional. Default is os.environ.get('GRID_NETWORK_URL', None).",
    default=os.environ.get("GRID_NETWORK_URL", None),
)

parser.add_argument(
    "--db_url",
    type=str,
    help="REDIS database server address",
    default=os.environ.get("REDISCLOUD_URL", ""),
)

parser.set_defaults(use_test_config=False)

if __name__ == "__main__":
    args = parser.parse_args()

    # Create app
    app = create_app(args.id, debug=False, database_url=args.db_url)

    # If using a Gateway URL start the connection
    if args.gateway_url is not None:
        if "http://" in args.host:
            node_address = "{}:{}".format(args.host, args.port)
        else:
            node_address = "http://{}:{}".format(args.host, args.port)
        requests.post(
            os.path.join(args.gateway_url, "join"),
            data=json.dumps({"node-id": args.id, "node-address": node_address}),
        )
    server = pywsgi.WSGIServer(("", args.port), app, handler_class=WebSocketHandler)
    server.serve_forever()
else:
    ## DEPLOYMENT MODE (we use gunicorn's worker to perform load balancing)

    # These environment variables must be set before starting the application.
    gateway_url = os.environ.get("GRID_NETWORK_URL", None)
    node_id = os.environ.get("ID", None)
    node_address = os.environ.get("ADDRESS", None)
    db_address = os.environ.get("REDISCLOUD_URL", None)

    # If using a Gateway URL start the connection
    if gateway_url:
        requests.post(
            os.path.join(gateway_url, "join"),
            data=json.dumps(
                {"node-id": node_id, "node-address": "{}".format(node_address)}
            ),
        )
    app = create_app(node_id, debug=False, database_url=db_address)
