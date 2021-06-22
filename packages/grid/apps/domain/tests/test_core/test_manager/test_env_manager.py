# third party
from bcrypt import checkpw
import pytest
from src.main.core.database import *
from src.main.core.database.environment.environment import Environment
from src.main.core.database.environment.user_environment import UserEnvironment
from src.main.core.exceptions import InvalidCredentialsError
from src.main.core.manager import EnvironmentManager
from src.main.core.manager import UserManager

user_role = ("User", False, False, False, False, False, False, False)
admin_role = ("Administrator", True, True, True, True, False, False, True)


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(User).delete()
        database.session.query(Role).delete()
        database.session.query(Group).delete()
        database.session.query(UserGroup).delete()
        database.session.query(Environment).delete()
        database.session.query(UserEnvironment).delete()
        database.session.commit()
    except:
        database.session.rollback()


def test_create_env_manager(database, cleanup):
    environments = EnvironmentManager(database)


def test_create_user_manager(database, cleanup):
    users = UserManager(database)


def test_register_new_environment(database, cleanup):
    environment = EnvironmentManager(database)

    new_env = environment.register(
        address="http://localhost:5000/",
        provider="AWS",
        instance_type="EC2",
        state=2,
    )

    assert new_env.address == "http://localhost:5000/"
    assert new_env.provider == "AWS"
    assert new_env.instance_type == "EC2"
    assert new_env.state == 2


def test_query_new_environment(database, cleanup):
    environment = EnvironmentManager(database)

    environment.register(
        address="http://localhost:5000/",
        provider="AWS",
        instance_type="EC2",
        state=2,
    )

    result = environment.query(address="http://localhost:5000/")[0]
    assert result.provider == "AWS"
    assert result.instance_type == "EC2"
    assert result.state == 2


def test_first_new_environment(database, cleanup):
    environment = EnvironmentManager(database)

    environment.register(
        address="http://localhost:5000/",
        provider="AWS",
        instance_type="EC2",
        state=2,
    )

    result = environment.first(address="http://localhost:5000/")
    assert result.provider == "AWS"
    assert result.instance_type == "EC2"
    assert result.state == 2
