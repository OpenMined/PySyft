# third party
import pytest
import torch as th

# syft absolute
from syft.core.common.message import SyftMessage
from syft.core.node.domain import Domain
from syft.core.node.domain.service import RequestStatus


@pytest.mark.asyncio
async def test_domain_creation() -> None:
    Domain(name="test domain")


@pytest.mark.asyncio
def test_domain_serde() -> None:

    domain_1 = Domain(name="domain 1")
    domain_1_client = domain_1.get_client()

    tensor = th.tensor([1, 2, 3])
    _ = tensor.send(domain_1_client)


@pytest.mark.asyncio
def test_domain_request_pending() -> None:
    domain_1 = Domain(name="remote domain")
    tensor = th.tensor([1, 2, 3])

    domain_1_client = domain_1.get_root_client()
    data_ptr_domain_1 = tensor.send(domain_1_client)

    domain_2 = Domain(name="my domain")

    data_ptr_domain_1.request(
        reason="I'd lke to see this pointer",
    )

    requested_object = data_ptr_domain_1.id_at_location

    # make request
    message_request_id = domain_1_client.requests.get_request_id_from_object_id(
        object_id=requested_object
    )

    # check status
    response = data_ptr_domain_1.check_access(
        node=domain_2, request_id=message_request_id
    )

    assert RequestStatus.Pending == response


@pytest.mark.slow
@pytest.mark.asyncio
def test_domain_request_denied() -> None:
    domain_1 = Domain(name="remote domain")
    tensor = th.tensor([1, 2, 3])

    domain_1_client = domain_1.get_root_client()
    data_ptr_domain_1 = tensor.send(domain_1_client)

    domain_2 = Domain(name="my domain")

    data_ptr_domain_1.request(reason="I'd lke to see this pointer")

    requested_object = data_ptr_domain_1.id_at_location

    # make request
    message_request_id = domain_1_client.requests.get_request_id_from_object_id(
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


@pytest.mark.asyncio
def test_domain_request_accepted() -> None:
    domain_1 = Domain(name="remote domain")
    tensor = th.tensor([1, 2, 3])

    domain_1_client = domain_1.get_root_client()
    data_ptr_domain_1 = tensor.send(domain_1_client)

    domain_2 = Domain(name="my domain")

    data_ptr_domain_1.request(reason="I'd lke to see this pointer")

    requested_object = data_ptr_domain_1.id_at_location

    message_request_id = domain_1_client.requests.get_request_id_from_object_id(
        object_id=requested_object
    )

    domain_1.requests[0].owner_client_if_available = domain_1_client
    domain_1.requests[0].accept()

    response = data_ptr_domain_1.check_access(
        node=domain_2, request_id=message_request_id
    )

    assert RequestStatus.Accepted == response


@pytest.mark.asyncio
def test_domain_is_for_me_exception() -> None:
    domain_1 = Domain(name="remote domain")

    with pytest.raises(Exception):
        msg = SyftMessage()
        domain_1.message_is_for_me(msg)
