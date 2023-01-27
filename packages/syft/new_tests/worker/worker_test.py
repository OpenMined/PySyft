# stdlib
from typing import Any

# third party
import numpy as np

# syft absolute
from syft.core.common.uid import UID
from syft.core.node.new.action_object import ActionObject
from syft.core.node.new.action_store import ActionStore
from syft.core.node.new.context import AuthedServiceContext
from syft.core.node.new.credentials import SIGNING_KEY_FOR
from syft.core.node.new.credentials import SyftSigningKey
from syft.core.node.new.credentials import SyftVerifyKey
from syft.core.node.new.user import User
from syft.core.node.new.user import UserUpdate
from syft.core.node.new.user_service import UserService
from syft.core.node.worker import Worker

# from syft.core.node.worker import Worker

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
    action_store = ActionStore()
    uid = UID()
    raw_data = np.array([1, 2, 3])
    test_object = ActionObject(syft_action_data=raw_data)

    set_result = action_store.set(
        uid=uid, credentials=test_signing_key, syft_object=test_object
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
    new_user = UserUpdate(
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
    print("new user", new_user)

    user = new_user.to(User)
    print("got a user", user)
    # assert user.id is not None # need to insert / update first
    assert user.email == "alice@bob.com"
    assert user.name == "Alice"
    assert user.hashed_password is not None
    assert user.salt is not None

    edit_user = user.to(UserUpdate)
    # assert edit_user.id is not None # need to insert / update first
    assert edit_user.email == "alice@bob.com"
    assert edit_user.name == "Alice"
    assert edit_user.password is None
    assert edit_user.password_verify is None
    assert not hasattr(edit_user, "signing_key")
    assert not hasattr(edit_user, "verify_key")


def test_user_service() -> None:
    test_signing_key = SyftSigningKey.from_string(test_signing_key_string)
    worker = Worker()
    user_collection = UserService()

    # create a user
    new_user = UserUpdate(
        email="alice@bob.com",
        name="Alice",
        password="letmein",
        password_verify="letmein",
    )

    # create a context
    context = AuthedServiceContext(node=worker, credentials=test_signing_key.verify_key)

    # call the create function
    user_view_result = user_collection.create(context=context, user_update=new_user)

    # get the result
    assert user_view_result.is_ok()
    user_view = user_view_result.ok()

    # we have a UID
    assert user_view.id is not None

    # we can query the same user again
    user_view_2_result = user_collection.view(context=context, uid=user_view.id)

    # the object matches
    assert user_view_2_result.is_ok()
    user_view_2 = user_view_2_result.ok()
    assert user_view == user_view_2


def test_syft_object_serde() -> None:
    # create a user
    new_user = UserUpdate(
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
    action_object = ActionObject(syft_action_data=raw_data)
    result = action_object + action_object
    x = result.syft_action_data
    y = raw_data * 2
    assert (x == y).all()


def test_action_object_hooks() -> None:
    raw_data = np.array([1, 2, 3])
    action_object = ActionObject(syft_action_data=raw_data)

    def pre_add(*args: Any, **kwargs: Any) -> Any:
        # double it
        new_value = args[0]
        new_value.syft_action_data = new_value.syft_action_data * 2
        return (new_value,), kwargs

    def post_add(result: Any) -> Any:
        # change return type to sum
        return sum(result.syft_action_data)

    action_object.syft_pre_hooks__["__add__"] = [pre_add]
    action_object.syft_post_hooks__["__add__"] = [post_add]

    result = action_object + action_object
    x = result.syft_action_data
    y = sum((raw_data * 2) + raw_data)
    assert y == 18
    assert x == y
