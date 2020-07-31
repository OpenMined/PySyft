import sys
from random import randint

import pytest

from src.users.role import Role
from .presets.role import role_metrics


@pytest.mark.parametrize(
    ("name, can_edit_settings, can_create_users," "can_edit_roles, can_manage_roles"),
    role_metrics,
)
def test_create_role_object(
    name,
    can_edit_settings,
    can_create_users,
    can_edit_roles,
    can_manage_roles,
    database,
):
    role = Role(
        name=name,
        can_edit_settings=can_edit_settings,
        can_create_users=can_create_users,
        can_edit_roles=can_edit_roles,
        can_manage_roles=can_manage_roles,
    )
    database.session.add(role)
    database.session.commit()
