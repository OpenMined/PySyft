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

    print(UserUpdate.__fields__)

    assert new_user.id is None
    assert new_user.email == "alice@bob.com"
    assert new_user.name == "Alice"
    assert new_user.password == "letmein"
    assert new_user.password_verify == "letmein"

    user = new_user.to(User)
    assert isinstance(user.id, UID)
    print("user", user)

    print("user and user update", type(user.email), type(new_user.email))

    result = user_stash.set(user)
    result2 = user_stash.get(user.id)
    print("result", type(result))
    print("result2", type(result2))

    assert result.email == result2.email
    assert result.name == result2.name
    assert result.hashed_password == result2.hashed_password
    assert result.salt == result2.salt
    assert result.signing_key == result2.signing_key
    assert result.verify_key == result2.verify_key
    assert result.role == result2.role
    assert result.institution == result2.institution
    assert result.website == result2.website
    assert result.created_at == result2.created_at

    assert result == result2
    assert False
