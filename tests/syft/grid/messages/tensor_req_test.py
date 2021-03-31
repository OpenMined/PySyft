# stdlib
from typing import Any
from typing import Dict

# syft absolute
import syft as sy
from syft.core.io.address import Address
from syft.grid.messages.request_messages import CreateRequestMessage
from syft.grid.messages.request_messages import CreateRequestResponse
from syft.grid.messages.request_messages import DeleteRequestMessage
from syft.grid.messages.request_messages import DeleteRequestResponse
from syft.grid.messages.request_messages import GetRequestMessage
from syft.grid.messages.request_messages import GetRequestResponse
from syft.grid.messages.request_messages import GetRequestsMessage
from syft.grid.messages.request_messages import GetRequestsResponse
from syft.grid.messages.request_messages import UpdateRequestMessage
from syft.grid.messages.request_messages import UpdateRequestResponse


def test_create_data_request_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content = {
        "dataset-id": "68a465aer3adf",
        "user-id": "user-id7",
        "request-type": "read",
    }
    msg = CreateRequestMessage(
        address=target,
        content=request_content,
        reply_to=node.address,
    )

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_create_data_request_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Request sent succesfully!"}
    msg = CreateRequestResponse(
        address=target,
        status_code=200,
        content=request_content,
    )

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_delete_data_request_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content = {"request_id": "f2a6as5d16fasd"}
    msg = DeleteRequestMessage(
        address=target,
        content=request_content,
        reply_to=node.address,
    )

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_delete_data_request_response_serde() -> None:
    target = Address(name="Alice")

    content = {"msg": "Data Request has been deleted!"}
    msg = DeleteRequestResponse(
        status_code=200,
        address=target,
        content=content,
    )

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_update_data_request_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {
        "request_id": "546a4d51",
        "dataset-id": "68a465aer3adf",
        "user-id": "user-id7",
        "request-type": "write",
    }
    msg = UpdateRequestMessage(
        address=target,
        content=content,
        reply_to=node.address,
    )

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_update_data_request_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Data request has been updated successfully!"}
    msg = UpdateRequestResponse(
        address=target,
        status_code=200,
        content=request_content,
    )

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_get_data_request_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {"request_id": "eqw9e4a5d846"}
    msg = GetRequestMessage(
        address=target,
        content=content,
        reply_to=node.address,
    )

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_get_data_request_response_serde() -> None:
    target = Address(name="Alice")

    content = {
        "request_id": "asfdaead131",
        "dataset-id": "68a465aer3adf",
        "user-id": "user-id7",
        "request-type": "read",
    }

    msg = GetRequestResponse(
        address=target,
        status_code=200,
        content=content,
    )

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_get_all_data_requests_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content: Dict[Any, Any] = {}
    msg = GetRequestsMessage(
        address=target,
        content=content,
        reply_to=node.address,
    )

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2


def test_get_all_data_requests_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {
        "workers": {
            "626sadaf631": {
                "dataset-id": "68a465aer3adf",
                "user-id": "user-id7",
                "request-type": "read",
            },
            "a84ew64wq6e": {
                "dataset-id": "98w4e54a6d",
                "user-id": "user-id9",
                "request-type": "write",
            },
        }
    }

    msg = GetRequestsResponse(
        address=target,
        status_code=200,
        content=request_content,
    )

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2
