import sys
import os
from typing import Callable
from multiprocessing import Process

import pytest

sys.path.append("../apps/network/src")

from apps.network.src.app import create_app as create_network
from apps.node.src.app import create_app as create_domain
from apps.worker.src.app import create_app as create_worker

network_instances = [5000]
domain_instances = [3000]
worker_instances = [3001, 3002, 3003]


def setup_node(method: Callable, port: int) -> None:
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    app = method()
    server = pywsgi.WSGIServer(("0.0.0.0", port), app, handler_class=WebSocketHandler)

    server.serve_forever()


@pytest.fixture(scope="session", autouse=True)
def init_instances():
    jobs = []

    for network_args in network_instances:
        p = Process(target=setup_node, args=(create_network, network_args))
        p.start()
        jobs.append(p)

    for domain_args in domain_instances:
        p = Process(target=setup_domain, args=(create_domain, domain_args))
        p.start()
        jobs.append(p)

    for worker_args in worker_instances:
        p = Process(target=setup_domain, args=(create_worker, worker_args))
        p.start()
        jobs.append(p)

    yield

    for job in jobs:
        job.terminate()
