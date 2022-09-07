# stdlib
from unittest.mock import patch

# third party
from nacl.signing import SigningKey
import pytest

# syft absolute
import syft as sy
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    CreateRoleMessage,
)
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    DeleteRoleMessage,
)
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    GetRoleMessage,
)
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    GetRoleResponse,
)
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    GetRolesMessage,
)
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    GetRolesResponse,
)
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    UpdateRoleMessage,
)
from syft.core.node.common.node_service.role_manager.role_manager_service import (
    RoleManagerService,
)
from syft.core.node.common.node_service.success_resp_message import (
    SuccessResponseMessage,
)

# TODO: Ionesio ,Rasswanth remove skip after adding tests for the NewRoleManger.


@pytest.mark.skip
def test_create_role_message(domain: sy.Domain) -> None:
    role_name = "New Role"
    user_key = SigningKey(domain.verify_key.encode())

    msg = CreateRoleMessage(
        address=domain.address,
        name=role_name,
        reply_to=domain.address,
    )

    reply = None
    with patch.object(domain.users, "can_edit_roles", return_value=True):
        reply = RoleManagerService.process(node=domain, msg=msg, verify_key=user_key)

    assert reply is not None
    assert isinstance(reply, SuccessResponseMessage) is True
    assert reply.resp_msg == "Role created successfully!"


@pytest.mark.skip
def test_update_role_message(domain: sy.Domain) -> None:
    domain.roles.register(**{"name": "RoleToUpdate"})
    role = domain.roles.first(**{"name": "RoleToUpdate"})
    new_name = "New Role Name"
    user_key = SigningKey(domain.verify_key.encode())

    msg = UpdateRoleMessage(
        address=domain.address,
        role_id=role.id,
        name=new_name,
        reply_to=domain.address,
    )

    reply = None
    with patch.object(domain.users, "can_edit_roles", return_value=True):
        reply = RoleManagerService.process(node=domain, msg=msg, verify_key=user_key)

    assert reply is not None
    assert reply.resp_msg == "Role updated successfully!"
    role_obj = domain.roles.first(**{"id": role.id})
    assert role_obj.name == new_name


@pytest.mark.skip
def test_get_role_message(domain: sy.Domain) -> None:
    role = domain.roles.first()
    user_key = SigningKey(domain.verify_key.encode())

    msg = GetRoleMessage(
        address=domain.address,
        role_id=role.id,
        reply_to=domain.address,
    )

    reply = None
    with patch.object(domain.users, "can_triage_requests", return_value=True):
        reply = RoleManagerService.process(node=domain, msg=msg, verify_key=user_key)

    assert reply is not None
    assert isinstance(reply, GetRoleResponse) is True
    assert reply.content is not None
    assert reply.content["name"] == role.name


@pytest.mark.skip
def test_get_roles_message(domain: sy.Domain) -> None:
    user_key = SigningKey(domain.verify_key.encode())

    msg = GetRolesMessage(
        address=domain.address,
        reply_to=domain.address,
    )

    reply = None
    with patch.object(domain.users, "can_triage_requests", return_value=True):
        reply = RoleManagerService.process(node=domain, msg=msg, verify_key=user_key)

    assert reply is not None
    assert isinstance(reply, GetRolesResponse) is True
    assert reply.content is not None
    assert type(reply.content) == list


@pytest.mark.skip
def test_del_role_manager(domain: sy.Domain) -> None:
    user_key = SigningKey(domain.verify_key.encode())

    # Create a dummy role to be deleted
    domain.roles.register(**{"name": "RoleToDelete"})
    role = domain.roles.first(**{"name": "RoleToDelete"})

    msg = DeleteRoleMessage(
        address=domain.address,
        reply_to=domain.address,
        role_id=role.id,
    )

    reply = None
    with patch.object(domain.users, "can_edit_roles", return_value=True):
        reply = RoleManagerService.process(node=domain, msg=msg, verify_key=user_key)

    assert reply is not None
    assert isinstance(reply, SuccessResponseMessage) is True
    assert reply.resp_msg == "Role has been deleted!"
