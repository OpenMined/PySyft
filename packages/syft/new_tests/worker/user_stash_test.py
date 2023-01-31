# syft absolute
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

    assert new_user.email == "alice@bob.com"
    assert new_user.name == "Alice"
    assert new_user.password == "letmein"
    assert new_user.password_verify == "letmein"

    user = new_user.to(User)

    print("user.id", user.id)

    result = user_stash.set(user)
    print("result", result)

    result2 = user_stash.get(user.id)
    print("result", result2)

    assert result == result2
