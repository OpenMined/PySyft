import pytest

import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(myPath + "/../src/")
from app import create_app
from main.core.database import db


@pytest.fixture(scope="function", autouse=True)
def app():
    db_path = "sqlite:///:memory:"
    return create_app(debug=True, test_config={"SQLALCHEMY_DATABASE_URI": db_path})


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(scope="function")
def database(app):
    test_db = db
    test_db.init_app(app)
    app.app_context().push()
    test_db.create_all()
    return test_db
