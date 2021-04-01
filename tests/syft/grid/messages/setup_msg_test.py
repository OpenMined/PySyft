# stdlib
from typing import Any
from typing import Dict

# syft absolute
import syft as sy
from syft import serialize
from syft.core.io.address import Address
from syft.grid.messages.setup_messages import CreateInitialSetUpMessage
from syft.grid.messages.setup_messages import CreateInitialSetUpResponse
from syft.grid.messages.setup_messages import GetSetUpMessage
from syft.grid.messages.setup_messages import GetSetUpResponse


def test_create_initial_setup_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content = {
        "settings": {
            "cloud-admin-token": "d84we35ad3a1d59a84sd9",
            "cloud-credentials": "<cloud-credentials.pem>",
            "infra": {"autoscaling": True, "triggers": {"memory": "50", "vCPU": "80"}},
        }
    }
    msg = CreateInitialSetUpMessage(
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


def test_create_initial_setup_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Initial setup registered successfully!"}
    msg = CreateInitialSetUpResponse(
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


def test_get_initial_setup_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content: Dict[Any, Any] = {}
    msg = GetSetUpMessage(
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

    content = {
        "settings": {
            "cloud-admin-token": "d84we35ad3a1d59a84sd9",
            "cloud-credentials": "<cloud-credentials.pem>",
            "infra": {"autoscaling": True, "triggers": {"memory": "50", "vCPU": "80"}},
        }
    }
    msg = GetSetUpResponse(
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
