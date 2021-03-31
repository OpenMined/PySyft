# stdlib
from typing import Any
from typing import Dict

# syft absolute
import syft as sy
from syft import serialize
from syft.core.io.address import Address
from syft.grid.messages.role_messages import CreateRoleMessage
from syft.grid.messages.role_messages import CreateRoleResponse
from syft.grid.messages.role_messages import DeleteRoleMessage
from syft.grid.messages.role_messages import DeleteRoleResponse
from syft.grid.messages.role_messages import GetRoleMessage
from syft.grid.messages.role_messages import GetRoleResponse
from syft.grid.messages.role_messages import GetRolesMessage
from syft.grid.messages.role_messages import GetRolesResponse
from syft.grid.messages.role_messages import UpdateRoleMessage
from syft.grid.messages.role_messages import UpdateRoleResponse


def test_create_role_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content = {
        "name": "Role Sample",
        "can_triage_results": True,
        "can_edit_settings": False,
        "can_create_users": False,
        "can_edit_roles": False,
        "can_manage_infrastructure": True,
    }
    msg = CreateRoleMessage(
        address=target,
        content=request_content,
        reply_to=node.address,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_create_role_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Role created succesfully!"}
    msg = CreateRoleResponse(
        address=target,
        status_code=200,
        content=request_content,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_delete_role_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content = {"role_id": "f2a6as5d16fasd"}
    msg = DeleteRoleMessage(
        address=target,
        content=request_content,
        reply_to=node.address,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_delete_role_response_serde() -> None:
    target = Address(name="Alice")

    content = {"msg": "Role has been deleted!"}
    msg = DeleteRoleResponse(
        status_code=200,
        address=target,
        content=content,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_update_role_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {
        "role_id": "9a4f9dasd6",
        "name": "Role Sample",
        "can_triage_results": True,
        "can_edit_settings": False,
        "can_create_users": False,
        "can_edit_roles": False,
        "can_manage_infrastructure": True,
    }
    msg = UpdateRoleMessage(
        address=target,
        content=content,
        reply_to=node.address,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_update_role_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Role has been updated successfully!"}
    msg = UpdateRoleResponse(
        address=target,
        status_code=200,
        content=request_content,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_get_role_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {"request_id": "eqw9e4a5d846"}
    msg = GetRoleMessage(
        address=target,
        content=content,
        reply_to=node.address,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_get_role_response_serde() -> None:
    target = Address(name="Alice")

    content = {
        "name": "Role Sample",
        "can_triage_results": True,
        "can_edit_settings": False,
        "can_create_users": False,
        "can_edit_roles": False,
        "can_manage_infrastructure": True,
    }

    msg = GetRoleResponse(
        address=target,
        status_code=200,
        content=content,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_get_all_roles_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content: Dict[Any, Any] = {}
    msg = GetRolesMessage(
        address=target,
        content=content,
        reply_to=node.address,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_get_all_roles_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {
        "workers": {
            "626sadaf631": {
                "name": "Role Sample",
                "can_triage_results": True,
                "can_edit_settings": False,
                "can_create_users": False,
                "can_edit_roles": False,
                "can_manage_infrastructure": True,
            },
            "a84ew64wq6e": {
                "name": "Test Sample",
                "can_triage_results": False,
                "can_edit_settings": True,
                "can_create_users": False,
                "can_edit_roles": False,
                "can_manage_infrastructure": False,
            },
        }
    }

    msg = GetRolesResponse(
        address=target,
        status_code=200,
        content=request_content,
    )

    blob = serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2
