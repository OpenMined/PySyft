import pytest
from src.app.main.users import Group

from .presets.group import group_metrics


@pytest.mark.parametrize(
    ("name"), group_metrics,
)
def test_create_group_object(
    name, database,
):
    new_group = Group(name=name)
    database.session.add(new_group)
    database.session.commit()
