import json
import os
import time
from multiprocessing import Process

import pytest
import syft
import torch
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

from . import GRID_NETWORK_PORT, IDS, PORTS


@pytest.fixture()
def start_proc():  # pragma: no cover
    """helper function for spinning up a websocket participant."""

    def _start_proc(participant, kwargs):
        def target():
            server = participant(**kwargs)
            server.start()

        p = Process(target=target)
        p.start()
        return p

    return _start_proc


@pytest.fixture(scope="session", autouse=True)
def node_infos():
    return zip(IDS, PORTS)


def setUpPyGrid(port, node_id):
    os.environ["SECRET_KEY"] = "Secretkeyhere"
    from apps.node.src.app import create_app

    db_path = "sqlite:///:memory:"
    app = create_app(
        debug=True,
        node_id=node_id,
        n_replica=1,
        test_config={"SQLALCHEMY_DATABASE_URI": db_path},
    )

    server = pywsgi.WSGIServer(("", int(port)), app, handler_class=WebSocketHandler)
    server.serve_forever()


def setup_network(port: int) -> None:
    """Setup gridnetwork.

    Args:
        port (int): port number

    Returns:
        None

    """

    from apps.network.src.app import create_app

    db_path = "sqlite:///:memory:"
    db_config = {"SQLALCHEMY_DATABASE_URI": db_path}

    app = create_app(debug=False, db_config=db_config)
    server = pywsgi.WSGIServer(("", port), app, handler_class=WebSocketHandler)
    server.serve_forever()


@pytest.fixture(scope="session", autouse=True)
def init_network_instance():

    p = Process(target=setup_network, args=(int(GRID_NETWORK_PORT),))
    p.start()
    time.sleep(2)

    yield

    p.terminate()


@pytest.fixture(scope="session", autouse=True)
def init_node_instances(node_infos):
    jobs = []

    # Init Grid Nodes
    for (node_id, port) in node_infos:
        p = Process(target=setUpPyGrid, args=(port, node_id))
        p.start()

        import requests

        requests.post(
            os.path.join("http://localhost:" + GRID_NETWORK_PORT, "join"),
            data=json.dumps(
                {"node-id": node_id, "node-address": "http://localhost:" + port}
            ),
        )

        time.sleep(2)
        jobs.append(p)

    yield

    for job in jobs:
        job.terminate()


@pytest.fixture(scope="session", autouse=True)
def hook():
    hook = syft.TorchHook(torch)
    return hook


@pytest.fixture(scope="function")
def connected_node(hook):
    nodes = {}
    for (node_id, port) in zip(IDS, PORTS):
        node = create_websocket_client(hook, port, node_id)
        nodes[node_id] = node

    yield nodes

    for node in nodes:
        nodes[node].close()
        time.sleep(0.1)


def create_websocket_client(hook, port, id):
    node = DataCentricFLClient(hook, "http://localhost:" + port, id=id)
    return node
