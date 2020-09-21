import pytest

from src.app.database.role import Role
from .presets.role import role_metrics


@pytest.mark.parametrize(
    ("name, can_edit_settings, can_create_users," "can_edit_roles, can_manage_nodes"),
    role_metrics,
)
def test_create_role_object(
    name,
    can_edit_settings,
    can_create_users,
    can_edit_roles,
    can_manage_nodes,
    database,
):
    role = Role(
        name=name,
        can_edit_settings=can_edit_settings,
        can_create_users=can_create_users,
        can_edit_roles=can_edit_roles,
        can_manage_nodes=can_manage_nodes,
    )
    database.session.add(role)
    database.session.commit()
