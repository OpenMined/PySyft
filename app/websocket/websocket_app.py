#!/bin/env python
"""
    Grid Node is a Socket/HTTP server used to manage / compute data remotely.
"""

from app import create_app, socketio
import sys
import requests
import json
import os
import argparse

# These environment variables must be set before starting the application.
gateway_url = os.environ.get("GRID_NETWORK_URL", None)
node_id = os.environ.get("ID", None)
node_address = os.environ.get("ADDRESS", None)
port = os.environ.get("PORT", None)
test_config = os.environ.get("TEST_CONFIG", None)

if test_config:
    app = create_app(debug=False, tst_config={"SQLALCHEMY_DATABASE_URI": test_config})
else:
    app = create_app(debug=False)


def check_args():
    parser = argparse.ArgumentParser(description="Run Grid Node application.")

    parser.add_argument(
        "--network",
        type=str,
        help="Network url (address used to join at grid network).",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the socket.io server, e.g. --port 8777.",
    )

    parser.add_argument("--addr", type=str, help="host for the connection.")

    parser.add_argument(
        "--id", type=str, help="name (id) of the grid node, e.g. --id alice."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = check_args()

    gateway_url = args.network if args.network else gateway_url
    port = args.port if args.port else port
    node_address = args.addr if args.addr else node_address
    node_id = args.id if args.id else node_id

    # Register request
    if gateway_url is not None:
        requests.post(
            os.path.join(gateway_url, "join"),
            data=json.dumps({"node-id": node_id, "node-address": node_address}),
        )

    socketio.run(app, host="0.0.0.0", port=port)
