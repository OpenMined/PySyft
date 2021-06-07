# stdlib
import os
import sys

# third party
import pytest

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(myPath + "/../src/")
# third party
from app import create_app
from main.core.database import db
import main.core.node
from main.core.node import GridDomain


@pytest.fixture(scope="function", autouse=True)
def app():
    args = {"start_local_db": True, "name": "OM Domain App"}
    args_obj = type("args", (object,), args)()
    return create_app(args_obj, testing=True)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def domain():
    return GridDomain(name="testing")


@pytest.fixture(scope="function")
def database(app):
    # TODO: Testing db should be used
    # but right now changes do not propagate
    # outside test suite

    # test_db = db
    # test_db.init_app(app)
    # app.app_context().push()
    # test_db.create_all()
    # return test_db
    return db
