# stdlib
from typing import Any
from typing import Dict

# syft absolute
import syft as sy
from syft import serialize
from syft.core.io.address import Address
from syft.grid.messages.infra_messages import CreateWorkerMessage
from syft.grid.messages.infra_messages import CreateWorkerResponse
from syft.grid.messages.infra_messages import DeleteWorkerMessage
from syft.grid.messages.infra_messages import DeleteWorkerResponse
from syft.grid.messages.infra_messages import GetWorkerMessage
from syft.grid.messages.infra_messages import GetWorkerResponse
from syft.grid.messages.infra_messages import GetWorkersMessage
from syft.grid.messages.infra_messages import GetWorkersResponse
from syft.grid.messages.infra_messages import UpdateWorkerMessage
from syft.grid.messages.infra_messages import UpdateWorkerResponse


def test_create_worker_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content = {
        "settings": {
            "instance-size": "t4g.medium",
            "vCPU": "2",
            "network-bandwith": "5Gbps",
            "vGPU": True,
        }
    }
    msg = CreateWorkerMessage(
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


def test_create_worker_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Worker Environment Created Successfully!"}
    msg = CreateWorkerResponse(
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


def test_delete_worker_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content = {"worker_id": "f2a6as5d16fasd"}
    msg = DeleteWorkerMessage(
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


def test_delete_worker_response_serde() -> None:
    target = Address(name="Alice")

    content = {"msg": "Worker Environment deleted Successfully!"}
    msg = DeleteWorkerResponse(
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


def test_update_worker_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {
        "worker-id": "eqw9e4a5d846",
        "settings": {
            "instance-size": "t4g.large",
            "vCPU": "2",
            "network-bandwith": "5Gbps",
            "vGPU": True,
        },
    }
    msg = UpdateWorkerMessage(
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


def test_update_worker_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Worker Environment updated successfully!"}
    msg = UpdateWorkerResponse(
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


def test_get_worker_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {"worker-id": "eqw9e4a5d846"}
    msg = GetWorkerMessage(
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


def test_get_worker_response_serde() -> None:
    target = Address(name="Alice")

    content = {
        "worker-id": "eqw9e4a5d846",
        "environment-name": "Heart Diseases Environment",
        "owner": "user-id7",
        "deployment-date": "05/12/2021",
    }

    msg = GetWorkerResponse(
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


def test_get_all_workers_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content: Dict[Any, Any] = {}
    msg = GetWorkersMessage(
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


def test_get_all_workers_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {
        "workers": {
            "626sadaf631": {
                "environment-name": "Heart Diseases Environment",
                "owner": "user-id7",
                "deployment-date": "05/12/2021",
            },
            "a84ew64wq6e": {
                "worker-id": "eqw9e4a5d846",
                "environment-name": "Brain Diseases Environment",
                "owner": "user-id8",
                "deployment-date": "15/12/2021",
            },
        }
    }

    msg = GetWorkersResponse(
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
