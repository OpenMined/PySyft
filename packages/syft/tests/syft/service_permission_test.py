# third party
import pytest

# syft absolute
from syft.client.api import SyftAPICall
from syft.types.errors import SyftException
from syft.types.syft_object import EXCLUDED_FROM_SIGNATURE


@pytest.fixture
def guest_mock_user(root_verify_key, user_stash, guest_user):
    user = user_stash.set(root_verify_key, guest_user).unwrap()
    assert user is not None
    yield user


def test_call_service_syftapi_with_permission(worker, guest_mock_user, update_user):
    user_id = guest_mock_user.id
    res = worker.root_client.api.services.user.update(
        uid=user_id,
        **{k: v for k, v in update_user if k not in EXCLUDED_FROM_SIGNATURE},
    )
    assert res


# this throws an AttributeError, maybe we want something more clear?
def test_call_service_syftapi_no_permission(guest_datasite_client):
    with pytest.raises(AttributeError):
        guest_datasite_client.api.services.user.get_all()


def test_directly_call_service_with_permission(worker, guest_mock_user, update_user):
    root_datasite_client = worker.root_client
    user_id = guest_mock_user.id
    api_call = SyftAPICall(
        server_uid=root_datasite_client.id,
        path="user.update",
        args=[],
        kwargs={"uid": user_id, **update_user},
    )
    signed_call = api_call.sign(root_datasite_client.api.signing_key)
    signed_result = root_datasite_client.api.connection.make_call(signed_call)
    result = signed_result.message.data
    assert result


def test_directly_call_service_no_permission(guest_datasite_client):
    api_call = SyftAPICall(
        server_uid=guest_datasite_client.id, path="user.get_all", args=[], kwargs={}
    )
    signed_call = api_call.sign(guest_datasite_client.api.signing_key)
    with pytest.raises(SyftException):
        guest_datasite_client.api.connection.make_call(signed_call)
