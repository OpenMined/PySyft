import torch as th
from syft.core.common import UID
from syft.core.node.domain import Domain
from syft.core.node.domain.service import (
    RequestStatus,
    RequestMessage,
    RequestAnswerResponse,
    RequestAnswerMessage,
)
from syft.core.io.address import Address
from syft import serialize, deserialize


def test_request_message() -> None:
    addr = Address()
    msg = RequestMessage(
        request_name="test request",
        request_description="test description",
        address=addr,
        owner_address=addr,
        object_id=UID(),
    )

    deserialized_obj = serialize(obj=msg)
    new_obj = deserialize(blob=deserialized_obj)

    assert msg.request_name == new_obj.request_name
    assert msg.request_description == new_obj.request_description
    assert msg.address == new_obj.address
    assert msg.owner_address == new_obj.owner_address
    assert msg.object_id == new_obj.object_id


def test_request_answer_message() -> None:

    addr = Address()

    msg = RequestAnswerMessage(request_id=UID(), address=addr, reply_to=addr)

    serialized = serialize(obj=msg)
    new_msg = deserialize(blob=serialized)

    assert msg.request_id == new_msg.request_id
    assert msg.address == new_msg.address
    assert msg.reply_to == new_msg.reply_to


def test_request_answer_response() -> None:

    addr = Address()

    msg = RequestAnswerResponse(
        request_id=UID(), address=addr, status=RequestStatus.Pending
    )

    serialized = serialize(obj=msg)
    new_msg = deserialize(blob=serialized)

    assert msg.request_id == new_msg.request_id
    assert msg.address == new_msg.address
    assert msg.status == new_msg.status


def test_domain_creation() -> None:
    Domain(name="test domain")


def test_domain_serde() -> None:

    domain_1 = Domain(name="domain 1")
    domain_1_client = domain_1.get_client()

    tensor = th.tensor([1, 2, 3])
    _ = tensor.send(domain_1_client)


# def test_domain_request_access():
#     domain_1 = Domain(name="remote domain")
#     tensor = th.tensor([1, 2, 3])
#     domain_1_client = domain_1.get_client()
#     data_ptr_domain_1 = tensor.send(domain_1_client)
#
#     domain_2 = Domain(name='my domain"')
#
#     data_ptr_domain_1.request_access(domain_2)
#
#     requested_object = data_ptr_domain_1.id_at_location
#     message_request_id = domain_2.requests.get_request_id_from_object_id(
#         requested_object
#     )
#
#     domain_1.set_request_status(message_request_id, RequestStatus.Accepted)
#
#     response = data_ptr_domain_1.check_access(
#         node=domain_2, request_id=message_request_id
#     )
#     print(response)

