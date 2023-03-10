# third party
import pytest

# syft absolute
from syft.core.common.uid import UID
from syft.core.node.new.credentials import SyftSigningKey
from syft.core.node.new.response import SyftSuccess
from syft.core.node.new.user import ServiceRole
from syft.core.node.new.user import User
from syft.core.node.new.user_stash import UserStash


@pytest.fixture
def user_stash(document_store):
    return UserStash(store=document_store)


def test_userstash_set(user_stash, guest_user):
    result = user_stash.set(guest_user)
    assert result.is_ok()

    created_user = result.ok()
    assert isinstance(created_user, User)
    assert guest_user == created_user


def test_userstash_set_duplicate(user_stash, guest_user):
    result = user_stash.set(guest_user)
    assert result.is_ok()

    result = user_stash.set(guest_user)
    assert result.is_err()

    assert "Duplication Key Error" in result.err()


def test_userstash_get_by_uid(user_stash, guest_user):
    result = user_stash.set(guest_user)
    assert result.is_ok()

    user = result.ok()
    assert user is not None

    result = user_stash.get_by_uid(uid=user.id)
    assert result.is_ok()

    searched_user = result.ok()

    assert user == searched_user

    random_uid = UID()
    result = user_stash.get_by_uid(uid=random_uid)

    searched_user = result.ok()
    assert result.is_ok()
    assert searched_user is None


def test_userstash_get_by_email(faker, user_stash, guest_user):
    result = user_stash.set(guest_user)
    assert result.is_ok()
    user = result.ok()
    assert user is not None

    result = user_stash.get_by_email(email=user.email)
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user

    random_email = faker.email()
    result = user_stash.get_by_email(email=random_email)
    searched_user = result.ok()
    assert result.is_ok()
    assert searched_user is None


def test_userstash_get_by_signing_key(user_stash, guest_user):
    result = user_stash.set(guest_user)
    assert result.is_ok()
    user = result.ok()
    assert user is not None

    result = user_stash.get_by_signing_key(signing_key=user.signing_key)
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user

    signing_key_as_str = str(user.signing_key)
    result = user_stash.get_by_signing_key(signing_key=signing_key_as_str)
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user

    random_singing_key = SyftSigningKey.generate()
    result = user_stash.get_by_signing_key(signing_key=random_singing_key)
    searched_user = result.ok()
    assert result.is_ok()
    assert searched_user is None


def test_userstash_get_by_verify_key(user_stash, guest_user):
    result = user_stash.set(guest_user)
    assert result.is_ok()
    user = result.ok()
    assert user is not None

    result = user_stash.get_by_verify_key(verify_key=user.verify_key)
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user

    verify_key_as_str = str(user.verify_key)
    result = user_stash.get_by_verify_key(verify_key=verify_key_as_str)
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user

    random_verify_key = SyftSigningKey.generate().verify_key
    result = user_stash.get_by_verify_key(verify_key=random_verify_key)
    searched_user = result.ok()
    assert result.is_ok()
    assert searched_user is None


def test_userstash_get_by_role(user_stash, guest_user):
    result = user_stash.set(guest_user)
    assert result.is_ok()
    user = result.ok()
    assert user is not None

    result = user_stash.get_by_role(role=ServiceRole.GUEST)
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user


def test_userstash_delete_by_uid(user_stash, guest_user):
    result = user_stash.set(guest_user)
    assert result.is_ok()
    user = result.ok()
    assert user is not None

    result = user_stash.delete_by_uid(uid=user.id)
    assert result.is_ok()
    response = result.ok()
    assert isinstance(response, SyftSuccess)
    assert str(user.id) in response.message

    result = user_stash.get_by_uid(uid=user.id)
    assert result.is_ok()
    searched_user = result.ok()
    assert searched_user is None
