# third party
from faker import Faker

# syft absolute
from syft.service.user.user import User
from syft.service.user.user import UserUpdate
from syft.service.user.user_stash import UserStash
from syft.types.credentials import SyftSigningKey
from syft.types.response import SyftSuccess
from syft.types.uid import UID
from syft.types.user_roles import ServiceRole


def add_mock_user(root_domain_client, user_stash: UserStash, user: User) -> User:
    # prepare: add mock data
    result = user_stash.partition.set(root_domain_client.credentials.verify_key, user)
    assert result.is_ok()

    user = result.ok()
    assert user is not None

    return user


def test_userstash_set(
    root_domain_client, user_stash: UserStash, guest_user: User
) -> None:
    result = user_stash.set(root_domain_client.credentials.verify_key, guest_user)
    assert result.is_ok()

    created_user = result.ok()
    assert isinstance(created_user, User)
    assert guest_user == created_user
    assert guest_user.id in user_stash.partition.data


def test_userstash_set_duplicate(
    root_domain_client, user_stash: UserStash, guest_user: User
) -> None:
    result = user_stash.set(root_domain_client.credentials.verify_key, guest_user)
    assert result.is_ok()

    original_count = len(user_stash.partition.data)

    result = user_stash.set(root_domain_client.credentials.verify_key, guest_user)
    assert result.is_err()

    assert "Duplication Key Error" in result.err()

    assert len(user_stash.partition.data) == original_count


def test_userstash_get_by_uid(
    root_domain_client, user_stash: UserStash, guest_user: User
) -> None:
    # prepare: add mock data
    user = add_mock_user(root_domain_client, user_stash, guest_user)

    result = user_stash.get_by_uid(
        root_domain_client.credentials.verify_key, uid=user.id
    )
    assert result.is_ok()

    searched_user = result.ok()

    assert user == searched_user

    random_uid = UID()
    result = user_stash.get_by_uid(
        root_domain_client.credentials.verify_key, uid=random_uid
    )
    assert result.is_ok()

    searched_user = result.ok()
    assert searched_user is None


def test_userstash_get_by_email(
    root_domain_client, faker: Faker, user_stash: UserStash, guest_user: User
) -> None:
    # prepare: add mock data
    user = add_mock_user(root_domain_client, user_stash, guest_user)

    result = user_stash.get_by_email(
        root_domain_client.credentials.verify_key, email=user.email
    )
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user

    random_email = faker.email()
    result = user_stash.get_by_email(
        root_domain_client.credentials.verify_key, email=random_email
    )
    searched_user = result.ok()
    assert result.is_ok()
    assert searched_user is None


def test_userstash_get_by_signing_key(
    root_domain_client, user_stash: UserStash, guest_user: User
) -> None:
    # prepare: add mock data
    user = add_mock_user(root_domain_client, user_stash, guest_user)

    result = user_stash.get_by_signing_key(
        root_domain_client.credentials.verify_key, signing_key=user.signing_key
    )
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user

    signing_key_as_str = str(user.signing_key)
    result = user_stash.get_by_signing_key(
        root_domain_client.credentials.verify_key, signing_key=signing_key_as_str
    )
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user

    random_singing_key = SyftSigningKey.generate()
    result = user_stash.get_by_signing_key(
        root_domain_client.credentials.verify_key, signing_key=random_singing_key
    )
    searched_user = result.ok()
    assert result.is_ok()
    assert searched_user is None


def test_userstash_get_by_verify_key(
    root_domain_client, user_stash: UserStash, guest_user: User
) -> None:
    # prepare: add mock data
    user = add_mock_user(root_domain_client, user_stash, guest_user)

    result = user_stash.get_by_verify_key(
        root_domain_client.credentials.verify_key, verify_key=user.verify_key
    )
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user

    verify_key_as_str = str(user.verify_key)
    result = user_stash.get_by_verify_key(
        root_domain_client.credentials.verify_key, verify_key=verify_key_as_str
    )
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user

    random_verify_key = SyftSigningKey.generate().verify_key
    result = user_stash.get_by_verify_key(
        root_domain_client.credentials.verify_key, verify_key=random_verify_key
    )
    searched_user = result.ok()
    assert result.is_ok()
    assert searched_user is None


def test_userstash_get_by_role(
    root_domain_client, user_stash: UserStash, guest_user: User
) -> None:
    # prepare: add mock data
    user = add_mock_user(root_domain_client, user_stash, guest_user)

    result = user_stash.get_by_role(
        root_domain_client.credentials.verify_key, role=ServiceRole.GUEST
    )
    assert result.is_ok()
    searched_user = result.ok()
    assert user == searched_user


def test_userstash_delete_by_uid(
    root_domain_client, user_stash: UserStash, guest_user: User
) -> None:
    # prepare: add mock data
    user = add_mock_user(root_domain_client, user_stash, guest_user)

    result = user_stash.delete_by_uid(
        root_domain_client.credentials.verify_key, uid=user.id
    )
    assert result.is_ok()
    response = result.ok()
    assert isinstance(response, SyftSuccess)
    assert str(user.id) in response.message

    result = user_stash.get_by_uid(
        root_domain_client.credentials.verify_key, uid=user.id
    )
    assert result.is_ok()
    searched_user = result.ok()
    assert searched_user is None


def test_userstash_update(
    root_domain_client, user_stash: UserStash, guest_user: User, update_user: UserUpdate
) -> None:
    # prepare: add mock data
    user = add_mock_user(root_domain_client, user_stash, guest_user)

    update_kwargs = update_user.to_dict(exclude_empty=True).items()

    for field_name, value in update_kwargs:
        setattr(user, field_name, value)

    result = user_stash.update(root_domain_client.credentials.verify_key, user=user)
    assert result.is_ok()
    updated_user = result.ok()
    assert isinstance(updated_user, User)
    assert user == updated_user
