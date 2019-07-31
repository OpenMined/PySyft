import pytest
import torch
from multiprocessing import Process
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
import os
import sys

# We need to add our rest api as a path since it is a separate application
# deployed on Heroku:
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../app/pg_rest_api")
from pg_app import create_app

import syft
from syft import TorchHook


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
def hook():
    hook = TorchHook(torch)
    return hook
