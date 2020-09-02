import pytest

from src.app import create_app, db


@pytest.fixture(scope="function", autouse=True)
def app():
    db_path = "sqlite:///:memory:"
    return create_app(debug=True, db_config={"SQLALCHEMY_DATABASE_URI": db_path})


@pytest.fixture(scope="function")
def database(app):
    test_db = db
    test_db.init_app(app)
    app.app_context().push()
    test_db.create_all()
    return test_db


@pytest.fixture
def client(app):
    return app.test_client()
