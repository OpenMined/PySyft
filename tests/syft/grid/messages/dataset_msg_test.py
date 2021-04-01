# stdlib
from typing import Any
from typing import Dict

# syft absolute
import syft as sy
from syft import serialize
from syft.core.io.address import Address
from syft.grid.messages.dataset_messages import CreateDatasetMessage
from syft.grid.messages.dataset_messages import CreateDatasetResponse
from syft.grid.messages.dataset_messages import DeleteDatasetMessage
from syft.grid.messages.dataset_messages import DeleteDatasetResponse
from syft.grid.messages.dataset_messages import GetDatasetMessage
from syft.grid.messages.dataset_messages import GetDatasetResponse
from syft.grid.messages.dataset_messages import GetDatasetsMessage
from syft.grid.messages.dataset_messages import GetDatasetsResponse
from syft.grid.messages.dataset_messages import UpdateDatasetMessage
from syft.grid.messages.dataset_messages import UpdateDatasetResponse


def test_create_dataset_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content = {
        "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
        "description": "Dataset Description",
        "tags": ["#x", "#data-sample"],
        "pointable": True,
        "read-permission": ["user-id1", "user-id2", "user-id3"],
        "write-permission": ["user-id1", "user-id5", "user-id9"],
    }
    msg = CreateDatasetMessage(
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


def test_create_dataset_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Dataset created succesfully!"}
    msg = CreateDatasetResponse(
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


def test_delete_dataset_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content = {"dataset_id": "f2a6as5d16fasd"}
    msg = DeleteDatasetMessage(
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


def test_delete_dataset_response_serde() -> None:
    target = Address(name="Alice")

    content = {"msg": "Dataset deleted successfully!"}
    msg = DeleteDatasetResponse(
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


def test_update_dataset_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {
        "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
        "description": "Dataset Description",
        "tags": ["#x", "#data-sample"],
        "pointable": True,
        "read-permission": ["user-id1", "user-id2", "user-id3"],
        "write-permission": ["user-id1", "user-id5", "user-id9"],
    }
    msg = UpdateDatasetMessage(
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


def test_update_dataset_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Dataset updated successfully!"}
    msg = UpdateDatasetResponse(
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


def test_get_dataset_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {"dataset_id": "eqw9e4a5d846"}
    msg = GetDatasetMessage(
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


def test_get_dataset_response_serde() -> None:
    target = Address(name="Alice")

    content = {
        "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
        "description": "Dataset Description",
        "tags": ["#x", "#data-sample"],
        "pointable": True,
        "read-permission": ["user-id1", "user-id2", "user-id3"],
        "write-permission": ["user-id1", "user-id5", "user-id9"],
    }

    msg = GetDatasetResponse(
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


def test_get_all_datasets_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content: Dict[Any, Any] = {}
    msg = GetDatasetsMessage(
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


def test_get_all_datasets_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {
        "workers": {
            "626sadaf631": {
                "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
                "description": "Dataset Description",
                "tags": ["#x", "#data-sample"],
                "pointable": True,
                "read-permission": ["user-id1", "user-id2", "user-id3"],
                "write-permission": ["user-id1", "user-id5", "user-id9"],
            },
            "a84ew64wq6e": {
                "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
                "description": "Dataset Description",
                "tags": ["#x", "#data-sample"],
                "pointable": False,
                "read-permission": ["user-id1", "user-id2", "user-id3"],
                "write-permission": [],
            },
        }
    }

    msg = GetDatasetsResponse(
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
