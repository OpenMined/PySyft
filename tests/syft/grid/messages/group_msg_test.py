# syft absolute
import syft as sy
from syft.core.io.address import Address
from syft.grid.messages.group_messages import CreateGroupMessage
from syft.grid.messages.group_messages import CreateGroupResponse
from syft.grid.messages.group_messages import DeleteGroupMessage
from syft.grid.messages.group_messages import DeleteGroupResponse
from syft.grid.messages.group_messages import GetAllGroupMessage
from syft.grid.messages.group_messages import GetAllGroupResponse
from syft.grid.messages.group_messages import GetGroupMessage
from syft.grid.messages.group_messages import GetGroupResponse
from syft.grid.messages.group_messages import UpdateGroupMessage
from syft.grid.messages.group_messages import UpdateGroupResponse


def test_create_group_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    request_content = {
            "group-name": "Heart diseases group", 
            "members": ["user-id1", "user-id2", "user-id3"],
            "data": [
                    { "id": "264632213",
                      "permissions": "read"},
                    { "id": "264613232",
                      "permissions": "write"},
                    { "id": "896632213",
                      "permissions": "read"},
            ]
    }
    msg = CreateGroupMessage(
        address=target,
        content= request_content,
        reply_to=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2

def test_create_group_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Group Created Successfully!"}
    msg = CreateGroupResponse(
        address=target,
        success=True,
        content= request_content,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2

def test_delete_group_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    request_content = {"group_id": "f2a6as5d16fasd"}
    msg = DeleteGroupMessage(
        address=target,
        content=request_content,
        reply_to=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2

def test_delete_group_response_serde() -> None:
    target = Address(name="Alice")

    content = {"msg": "Group deleted Successfully!"}
    msg = DeleteGroupResponse(
        success=True,
        address=target,
        content=content,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_update_group_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")

    content = {
            "group-id" : "eqw9e4a5d846",
            "group-name": "Brain diseases group", 
            "members": ["user-id1", "user-id2", "user-id3"],
            "data": [
                    { "id": "264632213",
                      "permissions": "read"},
                    { "id": "264613232",
                      "permissions": "write"},
                    { "id": "896632213",
                      "permissions": "read"},
            ]
    }
    msg = UpdateGroupMessage(
        address=target,
        content=content,
        reply_to=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_update_group_response_serde() -> None:
    target = Address(name="Alice")


    request_content = {"msg": "Group updated successfully!"}
    msg = UpdateGroupResponse(
        address=target,
        success=True,
        content=request_content,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_get_group_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")


    content = { "group-id" : "eqw9e4a5d846" }
    msg = GetGroupMessage(
        address=target,
        content=content,
        reply_to=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_get_group_response_serde() -> None:
    target = Address(name="Alice")


    content = {
            "group-id" : "eqw9e4a5d846",
            "group-name": "Heart diseases group", 
            "members": ["user-id1", "user-id2", "user-id3"],
            "data": [
                    { "id": "264632213",
                      "permissions": "read"},
                    { "id": "264613232",
                      "permissions": "write"},
                    { "id": "896632213",
                      "permissions": "read"},
            ]
    }
    
    msg = GetGroupResponse(
        address=target,
        success=True,
        content=content,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2

def test_get_all_groups_message_serde() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    target = Address(name="Alice")


    content = {}
    msg = GetAllGroupMessage(
        address=target,
        content=content,
        reply_to=bob_vm.address,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_get_all_groups_response_serde() -> None:
    target = Address(name="Alice")


    request_content = { "groups": 
        {
            "626sadaf631": {
                "group-name": "Heart diseases group", 
                "members": ["user-id1", "user-id2", "user-id3"],
                "data": [
                        { "id": "264632213",
                          "permissions": "read"},
                        { "id": "264613232",
                          "permissions": "write"},
                        { "id": "896632213",
                          "permissions": "read"},
                ]
            },
            "a84ew64wq6e": {
                "group-name": "Brain diseases group", 
                "members": ["user-id5", "user-id7", "user-id9"],
                "data": [
                        { "id": "26463afasd",
                          "permissions": "read"},
                        { "id": "264613dafeqwe",
                          "permissions": "write"},
                        { "id": "896632sdfsf",
                          "permissions": "read"},
                ]
            },
        }
    }

    msg = GetAllGroupResponse(
        address=target,
        success=True,
        content=request_content,
    )

    blob = msg.serialize()
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2