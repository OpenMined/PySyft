import pytest

import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(myPath + "/../src/")
from app import create_app
import main.core.node
from main.core.node import GridDomain


@pytest.fixture(scope="function", autouse=True)
def app():
    args = {"start_local_db": True, "name": "OM Domain App"}
    args_obj = type("args", (object,), args)()
    return create_app(args_obj)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def domain():
    return GridDomain(name="testing")
