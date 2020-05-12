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
def init_gateway():
    BASEDIR = os.path.dirname(os.path.dirname(__file__))

    def setUpGateway(port):
        os.environ["SECRET_KEY"] = "Secretkeyhere"
        from grid.app import create_app

        db_path = "sqlite:///" + BASEDIR + "/databaseGateway.db"
        app = create_app(
            debug=True, n_replica=1, test_config={"SQLALCHEMY_DATABASE_URI": db_path}
        )

        server = pywsgi.WSGIServer(
            ("", int(GATEWAY_PORT)), app, handler_class=WebSocketHandler
        )
        server.serve_forever()

    # Init Grid Gateway
    p = Process(target=setUpGateway, args=(GATEWAY_PORT,))
    p.start()
    time.sleep(5)

    yield

    p.terminate()


@pytest.fixture(scope="session", autouse=True)
def hook():
    hook = syft.TorchHook(torch)
    return hook
