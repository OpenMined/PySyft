import pytest

from grid.app import create_app


@pytest.fixture
def app():
    db_path = "sqlite:///:memory:"
    return create_app(test_config={"SQLALCHEMY_DATABASE_URI": db_path})


@pytest.fixture
def client(app):
    return app.test_client()
