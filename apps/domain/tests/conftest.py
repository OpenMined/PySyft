import pytest

import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(myPath + "/../src/")
from app import create_app

@pytest.fixture(scope="function", autouse=True)
def app():
    return create_app()


@pytest.fixture
def client(app):
    return app.test_client()

