import pytest
import torch
from multiprocessing import Process
import os
import sys

import syft
from syft.workers.node_client import NodeClient
from syft.grid.public_grid import PublicGridNetwork


from . import IDS, PORTS, GATEWAY_URL, GATEWAY_PORT
import time

import requests
import json

from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler


@pytest.fixture()
def start_proc():  # pragma: no cover
    """ helper function for spinning up a websocket participant """

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


@pytest.fixture(scope="session", autouse=True)
def init_pygrid_instances(node_infos):
    BASEDIR = os.path.dirname(os.path.dirname(__file__))

    def setUpPyGrid(port, node_id):
        os.environ["SECRET_KEY"] = "Secretkeyhere"
        from grid.app import create_app

        db_path = "sqlite:///:memory:"
        app = create_app(
            debug=True,
            node_id=node_id,
            n_replica=1,
            test_config={"SQLALCHEMY_DATABASE_URI": db_path},
        )

        server = pywsgi.WSGIServer(("", int(port)), app, handler_class=WebSocketHandler)
        server.serve_forever()

    jobs = []

    # Init Grid Nodes
    for (node_id, port) in node_infos:
        p = Process(target=setUpPyGrid, args=(port, node_id))
        p.start()
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
    node = NodeClient(hook, "http://localhost:" + port + "/", id=id)
    return node
