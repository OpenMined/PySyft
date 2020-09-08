import pytest
from src.app.main.database import Group

from .presets.group import group_metrics


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(Group).delete()
        database.session.commit()
    except:
        database.session.rollback()


@pytest.mark.parametrize(
    ("name"), group_metrics,
)
def test_create_group_object(name, database, cleanup):
    new_group = Group(name=name)
    database.session.add(new_group)
    database.session.commit()
