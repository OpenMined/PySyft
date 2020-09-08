import pytest
from src.app.main.database import UserGroup

from .presets.usergroup import usergroup_metrics


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(UserGroup).delete()
        database.session.commit()
    except:
        database.session.rollback()


@pytest.mark.parametrize(
    ("user", "group"), usergroup_metrics,
)
def test_create_usergroup_object(user, group, database, cleanup):
    new_usergroup = UserGroup(user=user, group=group)
    database.session.add(new_usergroup)
    database.session.commit()
