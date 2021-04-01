# stdlib
from typing import Any
from typing import Dict

# syft absolute
import syft as sy
from syft import serialize
from syft.core.io.address import Address
from syft.grid.messages.tensor_messages import CreateTensorMessage
from syft.grid.messages.tensor_messages import CreateTensorResponse
from syft.grid.messages.tensor_messages import DeleteTensorMessage
from syft.grid.messages.tensor_messages import DeleteTensorResponse
from syft.grid.messages.tensor_messages import GetTensorMessage
from syft.grid.messages.tensor_messages import GetTensorResponse
from syft.grid.messages.tensor_messages import GetTensorsMessage
from syft.grid.messages.tensor_messages import GetTensorsResponse
from syft.grid.messages.tensor_messages import UpdateTensorMessage
from syft.grid.messages.tensor_messages import UpdateTensorResponse


def test_create_tensor_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content = {
        "tensor": [1, 2, 3, 4, 5, 6],
        "description": "Tensor Description",
        "tags": ["#x", "#data-sample"],
        "pointable": True,
    }
    msg = CreateTensorMessage(
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


def test_create_tensor_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Tensor created succesfully!"}
    msg = CreateTensorResponse(
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


def test_delete_tensor_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content = {"tensor_id": "f2a6as5d16fasd"}
    msg = DeleteTensorMessage(
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


def test_delete_tensor_response_serde() -> None:
    target = Address(name="Alice")

    content = {"msg": "Tensor deleted successfully!"}
    msg = DeleteTensorResponse(
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


def test_update_tensor_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {
        "tensor_id": "546a4d51",
        "tensor": [1, 2, 3, 4, 5, 6],
        "description": "Tensor description",
        "tags": ["#x", "#data-sample"],
        "pointable": True,
    }
    msg = UpdateTensorMessage(
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


def test_update_tensor_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Tensor updated successfully!"}
    msg = UpdateTensorResponse(
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


def test_get_tensor_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {"tensor_id": "eqw9e4a5d846"}
    msg = GetTensorMessage(
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


def test_get_tensor_response_serde() -> None:
    target = Address(name="Alice")

    content = {
        "description": "Tensor description",
        "tags": ["#x", "#data-sample"],
    }

    msg = GetTensorResponse(
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


def test_get_all_tensors_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content: Dict[Any, Any] = {}
    msg = GetTensorsMessage(
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


def test_get_all_tensors_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {
        "workers": {
            "626sadaf631": {
                "tensor": [1, 2, 3, 4, 5, 6],
                "description": "Tensor description",
                "tags": ["#x", "#data-sample"],
                "pointable": True,
            },
            "a84ew64wq6e": {
                "tensor": [9, 8, 2, 3, 5, 6],
                "description": "Tensor sample description",
                "tags": ["#y", "#label-sample"],
                "pointable": True,
            },
        }
    }

    msg = GetTensorsResponse(
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
