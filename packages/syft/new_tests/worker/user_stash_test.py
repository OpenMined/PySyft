# third party
from result import Err

# syft absolute
from syft.core.common.uid import UID
from syft.core.node.new.document_store import DictDocumentStore
from syft.core.node.new.user import User
from syft.core.node.new.user import UserCreate
from syft.core.node.new.user import UserUpdate
from syft.core.node.new.user_stash import UserStash


def test_user_stash() -> None:
    store = DictDocumentStore()
    user_stash = UserStash(store=store)

    new_user = UserCreate(
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

    new_user.id = UID()
    # can only call set on User

    response = user_stash.set(new_user)
    assert isinstance(response, Err)

    result = user_stash.set(user)

    assert result == user

    result2 = user_stash.get_by_uid(user.id)

    result3 = user_stash.get_by_email(user.email)

    assert result2 == result3

    result4 = user_stash.get_by_signing_key(user.signing_key)

    assert result2 == result4

    result5 = user_stash.get_by_verify_key(user.verify_key)

    assert result2 == result5

    result6 = user_stash.find_one(
        **{"name": user.name, "email": user.email, "id": user.id}
    )
    result7 = user_stash.find_one(email=user.email)

    assert result6 == result
    assert result6 == result7

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

    result8 = user_stash.delete_by_uid(uid=user.id)
    assert result8 is True

    result9 = user_stash.get_by_uid(uid=user.id)
    assert len(result9) == 0

    result10 = user_stash.set(user)

    assert result10 == user

    result11 = user_stash.find_and_delete(**{"email": user.email})
    assert result11 is True

    update_user = UserUpdate(email="alice@bob.com", name="Bob", institution="OpenMined")
    result12 = user_stash.update(user=update_user.to(User))

    assert result12.is_ok() is False
    assert isinstance(result12, Err)

    new_user = UserCreate(
        email="alice@bob.com",
        name="Alice",
        password="letmein",
        password_verify="letmein",
    )

    user = new_user.to(User)
    result13 = user_stash.set(user)

    assert result13 == user

    update_user = UserUpdate(email="alice@bob.com", name="Bob", institution="OpenMined")
    result14 = user_stash.update(user=update_user.to(User))

    assert user.email == result14.email
    assert user.name == result14.name
    assert user.hashed_password == result13.hashed_password
    assert user.salt == result13.salt
    assert user.signing_key == result13.signing_key
    assert user.verify_key == result13.verify_key
    assert user.role == result13.role
    assert user.institution == result14.institution
    assert user.website == result13.website
    assert user.created_at == result13.created_at
