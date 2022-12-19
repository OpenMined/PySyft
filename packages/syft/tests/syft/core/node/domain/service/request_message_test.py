# stdlib
from unittest.mock import patch

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pytest
from pytest import mark
from pytest import raises

# syft absolute
from syft import Tensor
from syft import deserialize
from syft import serialize
from syft.core.adp.ledger_store import DictLedgerStore
from syft.core.common import UID
from syft.core.io.address import Address
from syft.core.node.common.node_manager.dict_store import DictStore
from syft.core.node.common.node_service.request_receiver.request_receiver_messages import (
    RequestMessage,
)
from syft.core.node.domain import Domain


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
        request_description="test description",
    )

    deserialized_obj = serialize(obj=msg)
    new_obj = deserialize(blob=deserialized_obj)

    assert msg.request_description == new_obj.request_description
    assert msg.address == new_obj.address
    assert msg.owner_address == new_obj.owner_address
    assert msg.object_id == new_obj.object_id
    assert msg.requester_verify_key == get_verify_key()


@pytest.mark.asyncio
@mark.parametrize("method_name", ["accept", "approve"])
def test_accept(method_name: str) -> None:
    node = Domain(
        name="remote domain", store_type=DictStore, ledger_store_type=DictLedgerStore
    )
    node_client = node.get_root_client()

    addr = Address()
    request = RequestMessage(
        object_id=UID(),
        address=addr,
        requester_verify_key=get_verify_key(),
        owner_address=addr,
        owner_client_if_available=node_client,
    )

    with patch.object(
        request.owner_client_if_available, "send_immediate_msg_without_reply"
    ) as mock_send_msg:
        getattr(request, method_name)()
        assert mock_send_msg.call_args[1]["msg"].address == node_client.address
        assert mock_send_msg.call_args[1]["msg"].accept is True
        assert mock_send_msg.call_args[1]["msg"].request_id == request.id


@pytest.mark.asyncio
def test_privacy_budget_and_obj_request():
    node = Domain(
        name="remote domain", store_type=DictStore, ledger_store_type=DictLedgerStore
    )
    root_client = node.get_root_client()

    node.initial_setup(
        signing_key=root_client.signing_key,
        first_superuser_name="Jane Doe",
        first_superuser_email="superuser@openmined.org",
        first_superuser_password="pwd",
        first_superuser_budget=5.55,
        domain_name="remote domain",
    )
    user_id = node.users.create_user_application(
        name="test", email="test@openmined.org", password="test", daa_pdf=b""
    )
    node.users.process_user_application(user_id, "accepted", root_client.verify_key)
    ds_client = node.get_client()
    ds_client.signing_key = SigningKey(
        node.users.first(name="test").private_key.encode("utf-8"), HexEncoder
    )
    ds_client.verify_key = ds_client.signing_key.verify_key

    Tensor([1, 2, 3, 4, 5]).send(root_client)

    ds_client.store[0].request(reason="Data Request Test")
    ds_client.request_budget(eps=1000, reason="Budget Request Test")

    assert root_client.requests[0].request_type == "data"
    assert root_client.requests[1].request_type == "budget"


@pytest.mark.asyncio
@mark.parametrize("method_name", ["deny", "reject", "withdraw"])
def test_deny(method_name: str) -> None:
    node = Domain(
        name="remote domain", store_type=DictStore, ledger_store_type=DictLedgerStore
    )
    node_client = node.get_root_client()

    addr = Address()
    request = RequestMessage(
        object_id=UID(),
        address=addr,
        requester_verify_key=get_verify_key(),
        owner_address=addr,
        owner_client_if_available=node_client,
    )

    with patch.object(
        request.owner_client_if_available, "send_immediate_msg_without_reply"
    ) as mock_send_msg:
        getattr(request, method_name)()
        assert mock_send_msg.call_args[1]["msg"].address == node_client.address
        assert mock_send_msg.call_args[1]["msg"].accept is False
        assert mock_send_msg.call_args[1]["msg"].request_id == request.id


def test_fail_accept_request_message() -> None:
    addr = Address()
    request = RequestMessage(
        object_id=UID(),
        address=addr,
        requester_verify_key=get_verify_key(),
        owner_address=addr,
    )

    with raises(Exception, match="No way to dispatch Accept Message."):
        request.accept()


def test_fail_deny_request_message() -> None:
    addr = Address()
    request = RequestMessage(
        object_id=UID(),
        address=addr,
        requester_verify_key=get_verify_key(),
        owner_address=addr,
    )

    with raises(Exception, match="No way to dispatch Deny Message."):
        request.deny()


def test_fail_process_request_service() -> None:
    addr = Address()
    request = RequestMessage(
        object_id=UID(),
        address=addr,
        requester_verify_key=get_verify_key(),
        owner_address=addr,
    )

    with raises(Exception, match="No way to dispatch Deny Message."):
        request.deny()
