import pytest
from src.app.main.users import UserGroup

from .presets.usergroup import usergroup_metrics


@pytest.mark.parametrize(
    ("user", "group"), usergroup_metrics,
)
def test_create_usergroup_object(user, group, database):
    new_usergroup = UserGroup(user=user, group=group,)
    database.session.add(new_usergroup)
    database.session.commit()
