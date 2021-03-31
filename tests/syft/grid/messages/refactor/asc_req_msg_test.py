from base_test import actual_test, abstract_test

# stdlib
from typing import Any
from typing import Dict

# syft absolute
import syft as sy
from syft import serialize
from syft.core.io.address import Address
from syft.grid.messages.association_messages import DeleteAssociationRequestMessage
from syft.grid.messages.association_messages import DeleteAssociationRequestResponse
from syft.grid.messages.association_messages import GetAssociationRequestMessage
from syft.grid.messages.association_messages import GetAssociationRequestResponse
from syft.grid.messages.association_messages import GetAssociationRequestsMessage
from syft.grid.messages.association_messages import GetAssociationRequestsResponse
from syft.grid.messages.association_messages import ReceiveAssociationRequestMessage
from syft.grid.messages.association_messages import ReceiveAssociationRequestResponse
from syft.grid.messages.association_messages import RespondAssociationRequestMessage
from syft.grid.messages.association_messages import RespondAssociationRequestResponse
from syft.grid.messages.association_messages import SendAssociationRequestMessage
from syft.grid.messages.association_messages import SendAssociationRequestResponse

def test_association_request_serde(node: sy.VirtualMachine) -> None:
    request_content = {"domain-name": "My-Domain", "domain-address": "http://url:5000"}
    abstract_test(SendAssociationRequestMessage, request_content, {"reply_to": node.address})
    
    response_content = {"msg": "Association Request Accepted status_codefully!"}
    abstract_test(SendAssociationRequestResponse, response_content, {"status_code": 200})

def test_receive_request_serde(node: sy.VirtualMachine) -> None:
    request_content: Dict[Any, Any] = {}
    abstract_test(ReceiveAssociationRequestMessage, request_content, {"reply_to": node.address})

    response_content: Dict[Any, Any] = {}
    abstract_test(ReceiveAssociationRequestResponse, request_content, {"status_code": 200})


def test_receive_request_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    request_content: Dict[Any, Any] = {}
    msg = ReceiveAssociationRequestMessage(
        address=target,
        content=request_content,
        reply_to=node.address,
    )

    actual_test(msg, target)

def test_receive_request_response_serde() -> None:
    target = Address(name="Alice")

    content: Dict[Any, Any] = {}
    msg = ReceiveAssociationRequestResponse(
        status_code=200,
        address=target,
        content=content,
    )

    actual_test(msg, target)

def test_respond_association_request_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {"association_request_id": "87564178", "status": "accept"}
    msg = RespondAssociationRequestMessage(
        address=target,
        content=content,
        reply_to=node.address,
    )

    actual_test(msg, target)

def test_respond_association_request_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {"msg": "Response registered status_codefully!"}
    msg = RespondAssociationRequestResponse(
        address=target,
        status_code=200,
        content=request_content,
    )

    actual_test(msg, target)

def test_delete_association_request_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {"association_request_id": "21656565"}
    msg = DeleteAssociationRequestMessage(
        address=target,
        content=content,
        reply_to=node.address,
    )

    actual_test(msg, target)

def test_delete_association_request_response_serde() -> None:
    target = Address(name="Alice")

    content = {"msg": "Association Request deleted status_codefully!"}
    msg = DeleteAssociationRequestResponse(
        address=target,
        status_code=200,
        content=content,
    )

    actual_test(msg, target)

def test_get_association_request_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content = {"association_request_id": "87564178"}
    msg = GetAssociationRequestMessage(
        address=target,
        content=content,
        reply_to=node.address,
    )

    actual_test(msg, target)

def test_get_association_request_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {
        "entity": "OpenMined",
        "entity-type": "Network",
        "status": "pending",
        "date": "05/12/2022",
    }
    msg = GetAssociationRequestResponse(
        address=target,
        status_code=200,
        content=request_content,
    )

    actual_test(msg, target)

def test_get_all_association_request_message_serde(node: sy.VirtualMachine) -> None:
    target = Address(name="Alice")

    content: Dict[Any, Any] = {}
    msg = GetAssociationRequestsMessage(
        address=target,
        content=content,
        reply_to=node.address,
    )

    actual_test(msg, target)

def test_get_all_association_request_response_serde() -> None:
    target = Address(name="Alice")

    request_content = {
        "association-requests": [
            {
                "entity": "OpenMined",
                "entity-type": "Network",
                "status": "pending",
                "date": "05/12/2022",
            },
            {
                "entity": "Hospital-A",
                "entity-type": "Domain",
                "status": "pending",
                "date": "09/10/2022",
            },
            {
                "entity": "OpenMined",
                "entity-type": "Network",
                "status": "pending",
                "date": "07/11/2022",
            },
        ]
    }
    msg = GetAssociationRequestsResponse(
        address=target,
        status_code=200,
        content=request_content,
    )

    actual_test(msg, target)