import pytest

from src import create_app


@pytest.fixture(scope="function", autouse=True)
def app():
    db_path = "sqlite:///:memory:"
    return create_app(debug=True, db_config={"SQLALCHEMY_DATABASE_URI": db_path})


@pytest.fixture
def client(app):
    return app.test_client()
