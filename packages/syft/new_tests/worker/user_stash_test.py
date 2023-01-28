# syft absolute
from syft.core.common.uid import UID
from syft.core.node.new.document_store import DictDocumentStore
from syft.core.node.new.user import User
from syft.core.node.new.user import UserUpdate
from syft.core.node.new.user_stash import UserStash


def test_user_stash() -> None:
    store = DictDocumentStore()
    user_stash = UserStash(store=store)

    new_user = UserUpdate(
        email="alice@bob.com",
        name="Alice",
        password="letmein",
        password_verify="letmein",
    )

    assert new_user.id is None
    assert new_user.email == "alice@bob.com"
    assert new_user.name == "Alice"
    assert new_user.password == "letmein"
    assert new_user.password_verify == "letmein"

    user = new_user.to(User)
    assert isinstance(user.id, UID)

    _ = user_stash.set(user)
    result2 = user_stash.get_by_uid(user.id)

    result3 = user_stash.get_by_email(user.email)

    assert result2 == result3

    assert user.email == result2.email
    assert user.name == result2.name
    assert user.hashed_password == result2.hashed_password
    assert user.salt == result2.salt
    assert user.signing_key == result2.signing_key
    assert user.verify_key == result2.verify_key
    assert user.role == result2.role
    assert user.institution == result2.institution
    assert user.website == result2.website
    assert user.created_at == result2.created_at

    assert user == result2
