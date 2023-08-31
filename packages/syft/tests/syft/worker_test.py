# stdlib
from typing import Any
from typing import Dict

# third party
from nacl.exceptions import BadSignatureError
import numpy as np
import pytest
from result import Ok

# syft absolute
import syft as sy
from syft.client.api import SignedSyftAPICall
from syft.client.api import SyftAPICall
from syft.node.credentials import SIGNING_KEY_FOR
from syft.node.credentials import SyftSigningKey
from syft.node.credentials import SyftVerifyKey
from syft.node.worker import Worker
from syft.service.action.action_object import ActionObject
from syft.service.action.action_store import DictActionStore
from syft.service.context import AuthedServiceContext
from syft.service.queue.queue_stash import QueueItem
from syft.service.response import SyftAttributeError
from syft.service.response import SyftError
from syft.service.user.user import User
from syft.service.user.user import UserCreate
from syft.service.user.user import UserView
from syft.service.user.user_service import UserService
from syft.types.uid import UID

test_signing_key_string = (
    "b7803e90a6f3f4330afbd943cef3451c716b338b17a9cf40a0a309bc38bc366d"
)

test_verify_key_string = (
    "08e5bcddfd55cdff0f7f6a62d63a43585734c6e7a17b2ffb3f3efe322c3cecc5"
)

test_signing_key_string_2 = (
    "8f4412396d3418d17c08a8f46592621a5d57e0daf1c93e2134c30f50d666801d"
)

test_verify_key_string_2 = (
    "833035a1c408e7f2176a0b0cd4ba0bc74da466456ea84f7ba4e28236e7e303ab"
)


def test_signing_key() -> None:
    # we should keep our representation in hex ASCII

    # first convert the string representation into a key
    test_signing_key = SyftSigningKey.from_string(test_signing_key_string)
    assert isinstance(test_signing_key, SyftSigningKey)

    # make sure it converts back to the same string
    assert str(test_signing_key) == test_signing_key_string

    # make a second one and verify that its equal
    test_signing_key_2 = SyftSigningKey.from_string(test_signing_key_string)
    assert test_signing_key == test_signing_key_2

    # get the derived verify key
    test_verify_key = test_signing_key.verify_key
    assert isinstance(test_verify_key, SyftVerifyKey)

    # make sure both types provide the verify key as a string
    assert test_verify_key_string == test_verify_key.verify
    assert test_verify_key_string == test_signing_key.verify

    # make sure that we don't print signing key but instead the verify key
    assert SIGNING_KEY_FOR in test_signing_key.__repr__()
    assert test_verify_key_string in test_signing_key.__repr__()

    # get another verify key from the same string and make sure its equal
    test_verify_key_2 = SyftVerifyKey.from_string(test_verify_key_string)
    assert test_verify_key == test_verify_key_2


def test_action_store() -> None:
    test_signing_key = SyftSigningKey.from_string(test_signing_key_string)
    action_store = DictActionStore()
    uid = UID()
    raw_data = np.array([1, 2, 3])
    test_object = ActionObject.from_obj(raw_data)

    set_result = action_store.set(
        uid=uid,
        credentials=test_signing_key,
        syft_object=test_object,
        has_result_read_permission=True,
    )
    assert set_result.is_ok()
    test_object_result = action_store.get(uid=uid, credentials=test_signing_key)
    assert test_object_result.is_ok()
    assert test_object == test_object_result.ok()

    test_verift_key_2 = SyftVerifyKey.from_string(test_verify_key_string_2)
    test_object_result_fail = action_store.get(uid=uid, credentials=test_verift_key_2)
    assert test_object_result_fail.is_err()
    assert "denied" in test_object_result_fail.err()


def test_user_transform() -> None:
    new_user = UserCreate(
        email="alice@bob.com",
        name="Alice",
        password="letmein",
        password_verify="letmein",
    )

    # assert new_user.id is None
    assert new_user.email == "alice@bob.com"
    assert new_user.name == "Alice"
    assert new_user.password == "letmein"
    assert new_user.password_verify == "letmein"

    user = new_user.to(User)
    # assert user.id is not None # need to insert / update first
    assert user.email == "alice@bob.com"
    assert user.name == "Alice"
    assert user.hashed_password is not None
    assert user.salt is not None

    edit_user = user.to(UserView)
    # assert edit_user.id is not None # need to insert / update first
    assert edit_user.email == "alice@bob.com"
    assert edit_user.name == "Alice"

    assert not hasattr(edit_user, "signing_key")


def test_user_service() -> None:
    test_signing_key = SyftSigningKey.from_string(test_signing_key_string)
    worker = Worker()
    user_service = worker.get_service(UserService)

    # create a user
    new_user = UserCreate(
        email="alice@bob.com",
        name="Alice",
        password="letmein",
        password_verify="letmein",
    )

    # create a context
    context = AuthedServiceContext(node=worker, credentials=test_signing_key.verify_key)

    # call the create function
    user_view = user_service.create(context=context, user_create=new_user)

    # get the result
    assert user_view is not None

    assert user_view.email == new_user.email
    assert user_view.name == new_user.name

    # we have a UID
    assert user_view.id is not None

    # we can query the same user again
    user_view_2 = user_service.view(context=context, uid=user_view.id)

    # the object matches
    assert user_view_2 is not None
    assert user_view == user_view_2


def test_syft_object_serde() -> None:
    # create a user
    new_user = UserCreate(
        email="alice@bob.com",
        name="Alice",
        password="letmein",
        password_verify="letmein",
    )
    # syft absolute
    import syft as sy

    ser = sy.serialize(new_user, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert new_user == de


def test_worker() -> None:
    worker = Worker()
    assert worker


def test_action_object_add() -> None:
    raw_data = np.array([1, 2, 3])
    action_object = ActionObject.from_obj(raw_data)
    result = action_object + action_object
    x = result.syft_action_data
    y = raw_data * 2
    assert (x == y).all()


def test_action_object_hooks() -> None:
    raw_data = np.array([1, 2, 3])
    action_object = ActionObject.from_obj(raw_data)

    def pre_add(context: Any, *args: Any, **kwargs: Any) -> Any:
        # double it
        new_value = args[0]
        new_value.syft_action_data = new_value.syft_action_data * 2
        return Ok((context, (new_value,), kwargs))

    def post_add(context: Any, name: str, new_result: Any) -> Any:
        # change return type to sum
        return Ok(sum(new_result))

    action_object._syft_pre_hooks__["__add__"] = [pre_add]
    action_object._syft_post_hooks__["__add__"] = [post_add]

    result = action_object + action_object
    x = result.syft_action_data
    y = sum((raw_data * 2) + raw_data)
    assert y == 18
    assert x == y

    action_object._syft_pre_hooks__["__add__"] = []
    action_object._syft_post_hooks__["__add__"] = []


def test_worker_serde() -> None:
    worker = Worker()
    ser = sy.serialize(worker, to_bytes=True)
    de = sy.deserialize(ser, from_bytes=True)

    assert de.signing_key == worker.signing_key
    assert de.id == worker.id


@pytest.mark.parametrize(
    "path, kwargs",
    [
        ("data_subject.get_all", {}),
        ("data_subject.get_by_name", {"name": "test"}),
        ("dataset.get_all", {}),
        ("dataset.search", {"name": "test"}),
        ("metadata", {}),
    ],
)
@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("n_processes", [0])
def test_worker_handle_api_request(
    path: str, kwargs: Dict, blocking: bool, n_processes: int
) -> None:
    node_uid = UID()
    test_signing_key = SyftSigningKey.from_string(test_signing_key_string)

    worker = Worker(
        name="test-domain-1",
        processes=n_processes,
        id=node_uid,
        signing_key=test_signing_key,
    )
    root_client = worker.root_client
    assert root_client.api is not None

    root_client.guest()

    # TODO: 🟡 Fix: root_client.guest is overriding root_client.
    root_client = worker.root_client

    api_call = SyftAPICall(
        node_uid=node_uid, path=path, args=[], kwargs=kwargs, blocking=blocking
    )
    # should fail on unsigned requests
    result = worker.handle_api_call(api_call).message.data
    assert isinstance(result, SyftError)

    signed_api_call = api_call.sign(root_client.api.signing_key)

    # should work on signed api calls
    result = worker.handle_api_call(signed_api_call).message.data
    assert not isinstance(result, SyftError)

    # Guest client should not have access to the APIs
    guest_signed_api_call = api_call.sign(root_client.api.signing_key)
    result = worker.handle_api_call(guest_signed_api_call).message
    assert not isinstance(result, SyftAttributeError)

    # should fail on altered requests
    bogus_api_call = signed_api_call
    bogus_api_call.serialized_message += b"hacked"

    result = worker.handle_api_call(bogus_api_call).message.data
    assert isinstance(result, SyftError)


@pytest.mark.parametrize(
    "path, kwargs",
    [
        ("data_subject.get_all", {}),
        ("data_subject.get_by_name", {"name": "test"}),
        ("dataset.get_all", {}),
        ("dataset.search", {"name": "test"}),
        ("metadata", {}),
    ],
)
@pytest.mark.parametrize("blocking", [False, True])
# @pytest.mark.parametrize("n_processes", [0, 1])
@pytest.mark.parametrize("n_processes", [0])
def test_worker_handle_api_response(
    path: str, kwargs: Dict, blocking: bool, n_processes: int
) -> None:
    test_signing_key = SyftSigningKey.from_string(test_signing_key_string)

    node_uid = UID()
    worker = Worker(
        name="test-domain-1",
        processes=n_processes,
        id=node_uid,
        signing_key=test_signing_key,
    )
    root_client = worker.root_client
    assert root_client.api is not None

    guest_client = root_client.guest()

    guest_client.register(
        name="Alice",
        email="alice@caltech.edu",
        password="abc123",
        password_verify="abc123",
    )

    # TODO: 🟡 Fix: root_client.guest is overriding root_client.
    root_client = worker.root_client

    call = SyftAPICall(
        node_uid=node_uid, path=path, args=[], kwargs=kwargs, blocking=blocking
    )
    signed_api_call = call.sign(root_client.credentials)

    # handle_api_call_with_unsigned_result should returned an unsigned result
    us_result = worker.handle_api_call_with_unsigned_result(signed_api_call)
    assert not isinstance(us_result, SignedSyftAPICall)

    # handle_api_call should return a signed result
    signed_result = worker.handle_api_call(signed_api_call)
    assert isinstance(signed_result, SignedSyftAPICall)

    # validation should work with the worker key
    root_client.credentials.verify_key.verify_key.verify(
        signed_result.serialized_message, signed_result.signature
    )
    # the validation should fail with the client key
    with pytest.raises(BadSignatureError):
        guest_client.credentials.verify_key.verify_key.verify(
            signed_result.serialized_message, signed_result.signature
        )

    # the signed result should be the same as the unsigned one
    result = signed_result.message.data
    assert isinstance(result, type(us_result))

    # the result should not be an error
    if path == "metadata":
        assert not isinstance(result, SyftError)
    elif not blocking and n_processes > 0:
        assert isinstance(result, QueueItem)
    else:
        assert not isinstance(result, SyftError)
