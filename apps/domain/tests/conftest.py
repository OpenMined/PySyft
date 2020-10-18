import pytest

import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(myPath + "/../src/")
from app import create_app


@pytest.fixture(scope="function", autouse=True)
def app():
    db_path = "sqlite:///databasenetwork.db"
    return create_app(test_config={"SQLALCHEMY_DATABASE_URI": db_path})


@pytest.fixture
def client(app):
    return app.test_client()
