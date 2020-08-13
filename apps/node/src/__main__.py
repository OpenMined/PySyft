#!/bin/env python

"""Grid Network is a Flask based application used to manage / monitor / control
and route grid workers remotely."""

import argparse
import os
import sys
import json
import requests

from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

from .app import create_app

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
    "--num_replicas",
    type=int,
    help="Number of replicas to provide fault tolerance to model hosting. If None no replica is used (aka num_replicas = 1). Default is os.environ.get('NUM_REPLICAS', None).",
    default=os.environ.get("NUM_REPLICAS", None),
)

parser.add_argument(
    "--start_local_db",
    dest="start_local_db",
    action="store_true",
    help="If this flag is used a SQLAlchemy DB URI is generated to use a local db.",
)

parser.add_argument(
    "--id", type=str, help="PyGrid Node ID.", default=os.environ.get("NODE_ID", None),
)

parser.set_defaults(use_test_config=False)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.start_local_db:
        db_path = "sqlite:///databasenetwork.db"
        app = create_app(
            node_id=args.id,
            debug=False,
            n_replica=args.num_replicas,
            test_config={"SQLALCHEMY_DATABASE_URI": db_path},
        )
    else:
        app = create_app(node_id=args.id, debug=False, n_replica=args.num_replicas)

    _network = args.network
    _address = "http://{}:{}".format(args.host, args.port)
    if _address and _network:
        requests.post(
            os.path.join(_network, "join"),
            data=json.dumps(
                {"node-id": args.id, "node-address": "{}".format(_address)}
            ),
        )

    server = pywsgi.WSGIServer(
        (args.host, args.port), app, handler_class=WebSocketHandler
    )
    server.serve_forever()
else:
    node_id = os.environ.get("NODE_ID", None)
    num_replicas = os.environ.get("N_REPLICAS", None)
    _address = os.environ.get("ADDRESS", None)
    _network = os.environ.get("NETWORK", None)
    if _address and _network:
        requests.post(
            os.path.join(_network, "join"),
            data=json.dumps(
                {"node-id": node_id, "node-address": "{}".format(_address)}
            ),
        )

    app = create_app(node_id=node_id, debug=False, n_replica=num_replicas)
