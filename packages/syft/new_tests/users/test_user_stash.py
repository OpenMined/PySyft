# third party
import pytest

# syft absolute
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
