import torch as th
from syft.core.common import UID
from syft.core.node.domain import Domain

from nacl.signing import SigningKey, VerifyKey

from syft.core.node.domain.service import (
    RequestStatus,
    RequestMessage,
    RequestAnswerResponse,
    RequestAnswerMessage,
)

from syft.core.io.address import Address
from syft import serialize, deserialize


def get_signing_key() -> SigningKey:
    # return a the signing key to use
    key = "e89ff2e651b42393b6ecb5956419088781309d953d72bd73a0968525a3a6a951"
    return SigningKey(bytes.fromhex(key))


def get_verify_key() -> VerifyKey:
    return get_signing_key().verify_key


def test_request_message() -> None:
    addr = Address()
    msg = RequestMessage(
        object_id=UID(),
        address=addr,
        requester_verify_key=get_verify_key(),
        owner_address=addr,
        request_name="test request",
        request_description="test description",
    )

    deserialized_obj = serialize(obj=msg)
    new_obj = deserialize(blob=deserialized_obj)

    assert msg.request_name == new_obj.request_name
    assert msg.request_description == new_obj.request_description
    assert msg.address == new_obj.address
    assert msg.owner_address == new_obj.owner_address
    assert msg.object_id == new_obj.object_id
    assert msg.requester_verify_key == get_verify_key()


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


def test_domain_request_access_pending() -> None:
    domain_1 = Domain(name="remote domain")
    tensor = th.tensor([1, 2, 3])

    domain_1_client = domain_1.get_root_client()
    data_ptr_domain_1 = tensor.send(domain_1_client)

    domain_2 = Domain(name="my domain")

    data_ptr_domain_1.request_access(
        request_name="My Request", reason="I'd lke to see this pointer"
    )

    requested_object = data_ptr_domain_1.id_at_location

    # make request
    message_request_id = domain_1_client.request_queue.get_request_id_from_object_id(
        object_id=requested_object
    )

    # check status
    response = data_ptr_domain_1.check_access(
        node=domain_2, request_id=message_request_id
    )

    assert RequestStatus.Pending == response


def test_domain_request_access_denied() -> None:
    domain_1 = Domain(name="remote domain")
    tensor = th.tensor([1, 2, 3])

    domain_1_client = domain_1.get_root_client()
    data_ptr_domain_1 = tensor.send(domain_1_client)

    domain_2 = Domain(name="my domain")

    data_ptr_domain_1.request_access(
        request_name="My Request", reason="I'd lke to see this pointer"
    )

    requested_object = data_ptr_domain_1.id_at_location

    # make request
    message_request_id = domain_1_client.request_queue.get_request_id_from_object_id(
        object_id=requested_object
    )

    # domain 1 client rejects request
    domain_1.requests[0].owner_client_if_available = domain_1_client
    domain_1.requests[0].deny()

    # check status
    response = data_ptr_domain_1.check_access(
        node=domain_2, request_id=message_request_id
    )

    assert RequestStatus.Rejected == response


def test_domain_request_access_accepted() -> None:
    domain_1 = Domain(name="remote domain")
    tensor = th.tensor([1, 2, 3])

    domain_1_client = domain_1.get_root_client()
    data_ptr_domain_1 = tensor.send(domain_1_client)

    domain_2 = Domain(name="my domain")

    data_ptr_domain_1.request_access(
        request_name="My Request", reason="I'd lke to see this pointer"
    )

    requested_object = data_ptr_domain_1.id_at_location

    message_request_id = domain_1_client.request_queue.get_request_id_from_object_id(
        object_id=requested_object
    )

    domain_1.requests[0].owner_client_if_available = domain_1_client
    domain_1.requests[0].accept()

    response = data_ptr_domain_1.check_access(
        node=domain_2, request_id=message_request_id
    )

    assert RequestStatus.Accepted == response
