# third party
from bcrypt import checkpw
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import pytest
from src import main
from src.main.core.database import *
from src.main.core.database import expand_user_object
from src.main.core.exceptions import InvalidCredentialsError
from src.main.core.exceptions import MissingRequestKeyError
from src.main.core.exceptions import UserNotFoundError
from src.main.core.manager import GroupManager
from src.main.core.manager import UserManager
from src.main.core.nodes.domain import GridDomain
from syft.grid.messages.user_messages import CreateUserMessage
from syft.grid.messages.user_messages import DeleteUserMessage
from syft.grid.messages.user_messages import GetUserMessage
from syft.grid.messages.user_messages import GetUsersMessage
from syft.grid.messages.user_messages import SearchUsersMessage
from syft.grid.messages.user_messages import UpdateUserMessage

owner = ("Owner", True, True, True, True, True, True, True)
user = ("User", False, False, False, False, False, False, False)
admin = ("Administrator", True, True, True, True, False, False, True)

# generate a signing key
generic_key = SigningKey.generate()


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(User).delete()
        database.session.query(Role).delete()
        database.session.query(Group).delete()
        database.session.query(UserGroup).delete()
        database.session.commit()
    except:
        database.session.rollback()


def __create_roles(database):
    owner_role = create_role(*owner)
    user_role = create_role(*user)
    admin_role = create_role(*admin)
    database.session.add(owner_role)
    database.session.add(user_role)
    database.session.add(admin_role)
    database.session.commit()


def build_syft_msg(node, msg_class, msg, key):
    content = {
        "address": node.address,
        "content": msg,
        "reply_to": node.address,
    }
    signed_msg = msg_class(**content).sign(signing_key=key)
    return node.recv_immediate_msg_with_reply(
        msg=signed_msg, raise_exception=True
    ).message


def __create_user_samples(node, users):
    owner_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }
    response = build_syft_msg(node, CreateUserMessage, owner_content, generic_key)

    assert response.status_code == 200

    user_content = {
        "email": "stduser@gmail.com",
        "password": "stduser123",
    }
    response = build_syft_msg(node, CreateUserMessage, user_content, generic_key)

    assert response.status_code == 200

    admin_content = {
        "email": "admin_user@gmail.com",
        "password": "admin_user",
        "role": "Administrator",
        "current_user": users.query(email="owner@gmail.com")[0].id,
    }
    response = build_syft_msg(node, CreateUserMessage, admin_content, generic_key)

    assert response.status_code == 200


def test_create_user_without_email(database, domain, cleanup):
    __create_roles(database)

    request_content = {
        "password": "owner123",
    }

    try:
        build_syft_msg(domain, CreateUserMessage, request_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "Invalid request payload, empty fields (email/password)!"


def test_create_user_without_password(database, domain, cleanup):
    __create_roles(database)

    request_content = {
        "email": "owner@gmail.com",
    }

    try:
        build_syft_msg(domain, CreateUserMessage, request_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "Invalid request payload, empty fields (email/password)!"


def test_create_first_user_msg(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)

    request_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }

    response = build_syft_msg(domain, CreateUserMessage, request_content, generic_key)

    # Check database
    assert len(users) == 1
    user = users.query(email="owner@gmail.com")[0]

    assert user.email == "owner@gmail.com"
    assert users.login(email="owner@gmail.com", password="owner123")
    assert users.role(user_id=user.id).name == "Owner"

    # Check message response
    assert response.status_code == 200
    assert database.session().query(Role).get(user.role).id == 1


def test_create_second_user(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)

    first_user_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }

    response = build_syft_msg(
        domain, CreateUserMessage, first_user_content, generic_key
    )

    # Check database
    assert len(users) == 1
    user = users.query(email="owner@gmail.com")[0]

    assert user.email == "owner@gmail.com"
    assert users.login(email="owner@gmail.com", password="owner123")
    assert users.role(user_id=user.id).name == "Owner"
    assert users.role(user_id=user.id).can_create_users
    assert users.role(user_id=user.id).can_triage_requests

    # Check message response
    assert response.status_code == 200

    second_user_content = {
        "email": "stduser@gmail.com",
        "password": "stduser123",
    }

    response = build_syft_msg(
        domain, CreateUserMessage, second_user_content, generic_key
    )

    # Check database
    assert len(users) == 2
    user = users.query(email="stduser@gmail.com")[0]

    assert user.email == "stduser@gmail.com"
    assert users.login(email="stduser@gmail.com", password="stduser123")
    assert users.role(user_id=user.id).name == "User"
    assert not users.role(user_id=user.id).can_create_users
    assert not users.role(user_id=user.id).can_triage_requests

    # Check message response
    assert response.status_code == 200


def test_create_second_user_with_invalid_role_name(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)

    owner_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }

    response = build_syft_msg(domain, CreateUserMessage, owner_content, generic_key)

    # Check database
    assert len(users) == 1
    user = users.query(email="owner@gmail.com")[0]

    assert user.email == "owner@gmail.com"
    assert users.login(email="owner@gmail.com", password="owner123")
    assert users.role(user_id=user.id).name == "Owner"
    assert users.role(user_id=user.id).can_create_users
    assert users.role(user_id=user.id).can_triage_requests

    # Check message response
    assert response.status_code == 200

    owner_id = str(users.query(email="owner@gmail.com")[0].id)

    second_user_content = {
        "email": "admin_user@gmail.com",
        "password": "admin_user123",
        "role": "Random Role",
        "current_user": owner_id,
    }

    try:
        build_syft_msg(domain, CreateUserMessage, second_user_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "Role ID not found!"

    # Check database
    assert len(users) == 1


def test_create_second_user_with_owner_role_name(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)

    owner_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }

    response = build_syft_msg(domain, CreateUserMessage, owner_content, generic_key)

    # Check database
    assert len(users) == 1
    user = users.query(email="owner@gmail.com")[0]

    assert user.email == "owner@gmail.com"
    assert users.login(email="owner@gmail.com", password="owner123")
    assert users.role(user_id=user.id).name == "Owner"
    assert users.role(user_id=user.id).can_create_users
    assert users.role(user_id=user.id).can_triage_requests

    # Check message response
    assert response.status_code == 200

    owner_id = str(users.query(email="owner@gmail.com")[0].id)

    second_user_content = {
        "email": "admin_user@gmail.com",
        "password": "admin_user123",
        "role": "Owner",
        "current_user": owner_id,
    }

    try:
        build_syft_msg(domain, CreateUserMessage, second_user_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == 'You can\'t create a new User with "Owner" role!'

    # Check database
    assert len(users) == 1


def test_create_second_user_with_role_and_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)

    request_content = {
        "email": "owner@gmail.com",
        "password": "owner123",
    }

    response = build_syft_msg(domain, CreateUserMessage, request_content, generic_key)

    # Check database
    assert len(users) == 1
    user = users.query(email="owner@gmail.com")[0]

    assert user.email == "owner@gmail.com"
    assert users.login(email="owner@gmail.com", password="owner123")
    assert users.role(user_id=user.id).name == "Owner"
    assert users.role(user_id=user.id).can_create_users
    assert users.role(user_id=user.id).can_triage_requests

    # Check message response
    assert response.status_code == 200

    owner_id = str(users.query(email="owner@gmail.com")[0].id)

    second_user_content = {
        "email": "admin_user@gmail.com",
        "password": "admin_user123",
        "role": "Administrator",
        "current_user": owner_id,
    }

    response = build_syft_msg(
        domain, CreateUserMessage, second_user_content, generic_key
    )

    # Check database
    assert len(users) == 2
    user = users.query(email="admin_user@gmail.com")[0]

    assert user.email == "admin_user@gmail.com"
    assert users.login(email="admin_user@gmail.com", password="admin_user123")
    assert users.role(user_id=user.id).name == "Administrator"
    assert users.role(user_id=user.id).can_create_users == True
    assert users.role(user_id=user.id).can_triage_requests == True

    # Check message response
    assert response.status_code == 200


def test_get_user_with_permissions(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    owner_user_id = users.query(email="owner@gmail.com")[0].id
    msg_content = {"user_id": owner_user_id, "current_user": owner_user_id}
    response = build_syft_msg(domain, GetUserMessage, msg_content, generic_key)
    assert response.status_code == 200
    retrieved_user = response.content
    assert retrieved_user["email"] == "owner@gmail.com"
    assert retrieved_user["id"] == 1
    assert retrieved_user["role"] == 1

    std_user_id = users.query(email="stduser@gmail.com")[0].id
    msg_content = {"user_id": std_user_id, "current_user": owner_user_id}
    response = build_syft_msg(domain, GetUserMessage, msg_content, generic_key)
    assert response.status_code == 200
    retrieved_user = response.content
    assert retrieved_user["email"] == "stduser@gmail.com"
    assert retrieved_user["id"] == 2
    assert retrieved_user["role"] == 2

    admin_user_id = users.query(email="admin_user@gmail.com")[0].id
    msg_content = {"user_id": admin_user_id, "current_user": owner_user_id}
    response = build_syft_msg(domain, GetUserMessage, msg_content, generic_key)
    assert response.status_code == 200
    retrieved_user = response.content
    assert retrieved_user["email"] == "admin_user@gmail.com"
    assert retrieved_user["id"] == 3
    assert retrieved_user["role"] == 3


def test_get_user_without_permissions(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    owner_user_id = users.query(email="owner@gmail.com")[0].id
    std_user_id = users.query(email="stduser@gmail.com")[0].id
    msg_content = {"user_id": owner_user_id, "current_user": std_user_id}

    try:
        build_syft_msg(domain, GetUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "You're not allowed to get User information!"


def test_get_user_with_invalid_fields(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    owner_user_id = users.query(email="owner@gmail.com")[0].id
    msg_content = {"current_user": owner_user_id}

    try:
        build_syft_msg(domain, GetUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "User not found!"


def test_get_all_users_with_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    owner_user_id = users.query(email="owner@gmail.com")[0].id
    msg_content = {"current_user": owner_user_id}
    response = build_syft_msg(domain, GetUsersMessage, msg_content, generic_key)
    assert response.status_code == 200
    assert len(response.content) == 3

    retrieved_user = response.content[0]
    assert retrieved_user["email"] == "owner@gmail.com"
    assert retrieved_user["id"] == 1
    assert retrieved_user["role"] == 1

    retrieved_user = response.content[1]
    assert retrieved_user["email"] == "stduser@gmail.com"
    assert retrieved_user["id"] == 2
    assert retrieved_user["role"] == 2

    retrieved_user = response.content[2]
    assert retrieved_user["email"] == "admin_user@gmail.com"
    assert retrieved_user["id"] == 3
    assert retrieved_user["role"] == 3


def test_get_all_users_without_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    std_user_id = users.query(email="stduser@gmail.com")[0].id
    msg_content = {"current_user": std_user_id}
    try:
        response = build_syft_msg(domain, GetUsersMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "You're not allowed to get User information!"


def test_delete_user_with_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    owner_user_id = users.query(email="owner@gmail.com")[0].id
    std_user_id = users.query(email="stduser@gmail.com")[0].id
    msg_content = {"user_id": std_user_id, "current_user": owner_user_id}
    response = build_syft_msg(domain, DeleteUserMessage, msg_content, generic_key)
    assert response.status_code == 200
    assert {"msg": "User deleted successfully!"} == {
        "msg": "User deleted successfully!"
    }


def test_delete_user_without_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    std_user_id = users.query(email="stduser@gmail.com")[0].id
    msg_content = {"user_id": std_user_id, "current_user": std_user_id}
    try:
        response = build_syft_msg(domain, DeleteUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "You're not allowed to delete this user information!"


def test_delete_owner_with_admin_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    owner_user_id = users.query(email="owner@gmail.com")[0].id
    admin_user_id = users.query(email="admin_user@gmail.com")[0].id
    msg_content = {"user_id": owner_user_id, "current_user": admin_user_id}

    try:
        response = build_syft_msg(domain, DeleteUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "You're not allowed to delete this user information!"


def test_delete_user_with_invalid_user_id(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    owner_user_id = users.query(email="owner@gmail.com")[0].id
    msg_content = {"user_id": "10", "current_user": owner_user_id}

    try:
        build_syft_msg(domain, DeleteUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "User not found!"


def test_search_users_without_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    std_user_id = users.query(email="stduser@gmail.com")[0].id
    msg_content = {"email": "owner@gmail.com", "current_user": std_user_id}
    try:
        build_syft_msg(domain, DeleteUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "You're not allowed to delete this user information!"


def test_search_unique_query(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    owner_user_id = users.query(email="owner@gmail.com")[0].id

    msg_content = {"email": "stduser@gmail.com", "current_user": owner_user_id}
    response = build_syft_msg(domain, SearchUsersMessage, msg_content, generic_key)
    assert response.status_code == 200

    search_result = response.content
    print("Result: ", search_result)
    assert len(search_result) == 1
    assert search_result[0]["email"] == "stduser@gmail.com"
    assert search_result[0]["role"] == 2

    msg_content = {"role": 2, "current_user": owner_user_id}
    response = build_syft_msg(domain, SearchUsersMessage, msg_content, generic_key)
    assert response.status_code == 200

    search_result = response.content
    assert len(search_result) == 1
    assert search_result[0]["email"] == "stduser@gmail.com"
    assert search_result[0]["role"] == 2


def test_search_multiple_query_parameters(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    owner_user_id = users.query(email="owner@gmail.com")[0].id

    # Correct
    msg_content = {
        "email": "stduser@gmail.com",
        "role": 2,
        "current_user": owner_user_id,
    }
    response = build_syft_msg(domain, SearchUsersMessage, msg_content, generic_key)
    assert response.status_code == 200

    search_result = response.content
    assert len(search_result) == 1
    assert search_result[0]["email"] == "stduser@gmail.com"
    assert search_result[0]["role"] == 2

    # Wrong
    msg_content = {
        "email": "stduser@gmail.com",
        "role": 1,
        "current_user": owner_user_id,
    }
    response = build_syft_msg(domain, SearchUsersMessage, msg_content, generic_key)
    assert response.status_code == 200
    assert response.content == {}


def test_search_query_mutual_parameter_values(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    new_user_content = {
        "email": "newstduser@gmail.com",
        "password": "newstduser123",
    }
    response = build_syft_msg(domain, CreateUserMessage, new_user_content, generic_key)
    assert response.status_code == 200

    owner_user_id = users.query(email="owner@gmail.com")[0].id

    # Correct
    msg_content = {"role": 2, "current_user": owner_user_id}
    response = build_syft_msg(domain, SearchUsersMessage, msg_content, generic_key)
    assert response.status_code == 200

    search_result = response.content
    assert len(search_result) == 2
    assert search_result[0]["email"] == "stduser@gmail.com"
    assert search_result[0]["role"] == 2

    assert search_result[1]["email"] == "newstduser@gmail.com"
    assert search_result[1]["role"] == 2


def test_update_invalid_user(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    # Correct
    owner_user_id = users.query(email="owner@gmail.com")[0].id
    msg_content = {
        "user_id": "10",
        "email": "testing@email.com",
        "current_user": owner_user_id,
    }
    try:
        build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "User not found!"


def test_update_empty_body(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    # Correct
    owner_user_id = users.query(email="owner@gmail.com")[0].id
    msg_content = {
        "user_id": "56",
        "current_user": owner_user_id,
    }
    try:
        build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "Missing json fields ( email,password,role,groups )"


def test_update_user_without_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    # Correct
    std_user_id = users.query(email="stduser@gmail.com")[0].id
    owner_id = users.query(email="owner@gmail.com")[0].id
    msg_content = {
        "user_id": owner_id,
        "email": "testing@email.com",
        "current_user": std_user_id,
    }
    try:
        build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "You're not allowed to change other user data!"


def test_update_its_own_user_email(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    # Correct
    std_user_id = users.query(email="stduser@gmail.com")[0].id
    msg_content = {
        "user_id": std_user_id,
        "email": "mynewemail@email.com",
        "current_user": std_user_id,
    }
    response = build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
    assert response.status_code == 200
    assert response.content == {"message": "User updated successfully!"}

    # Old email will no longer exist
    with pytest.raises(UserNotFoundError) as exc:
        users.query(email="stduser@gmail.com")

    assert len(users.query(email="mynewemail@email.com")) == 1


def test_update_its_own_user_password(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    # Correct
    std_user_id = users.query(email="stduser@gmail.com")[0].id
    msg_content = {
        "user_id": std_user_id,
        "password": "my_new_password",
        "current_user": std_user_id,
    }
    response = build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
    assert response.status_code == 200
    assert response.content == {"message": "User updated successfully!"}

    assert users.login(email="stduser@gmail.com", password="my_new_password")


def test_update_role_without_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    # Correct
    std_user_id = users.query(email="stduser@gmail.com")[0].id
    msg_content = {
        "user_id": std_user_id,
        "role": "2",
        "current_user": std_user_id,
    }

    try:
        build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "You're not allowed to change User roles!"


def test_update_role_with_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    # Correct
    std_user_id = users.query(email="stduser@gmail.com")[0].id
    owner_user_id = users.query(email="owner@gmail.com")[0].id
    msg_content = {
        "user_id": std_user_id,
        "role": "3",
        "current_user": owner_user_id,
    }
    response = build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
    assert response.status_code == 200
    assert response.content == {"message": "User updated successfully!"}

    user = users.query(email="stduser@gmail.com")[0]
    assert user.email == "stduser@gmail.com"
    assert users.role(user_id=user.id).name == "Administrator"


def test_update_to_invalid_role(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    # Correct
    std_user_id = users.query(email="stduser@gmail.com")[0].id
    owner_user_id = users.query(email="owner@gmail.com")[0].id
    msg_content = {
        "user_id": std_user_id,
        "role": "InvalidRole",
        "current_user": owner_user_id,
    }
    try:
        build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "Role ID not found!"


def test_update_to_owner_role(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    __create_user_samples(domain, users)

    # Correct
    std_user_id = users.query(email="stduser@gmail.com")[0].id
    admin_user = users.query(email="admin_user@gmail.com")[0].id
    msg_content = {
        "user_id": std_user_id,
        "role": "1",
        "current_user": admin_user,
    }
    try:
        build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "You can't change it to Owner role!"


def test_update_user_group_without_permission(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    groups = GroupManager(database)
    __create_user_samples(domain, users)

    # Create Group Samples
    group1 = groups.register(name="Group A")

    # Correct
    std_user_id = users.query(email="stduser@gmail.com")[0].id
    msg_content = {
        "user_id": std_user_id,
        "groups": [group1.id],
        "current_user": std_user_id,
    }
    try:
        build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
        pytest.fail("We shouldn't execute this line!")
    except Exception as e:
        assert str(e) == "You're not allowed to change User groups!"


def test_update_user_group(database, domain, cleanup):
    __create_roles(database)
    users = UserManager(database)
    groups = GroupManager(database)
    __create_user_samples(domain, users)

    # Create Group Samples
    group1 = groups.register(name="Group A")
    group2 = groups.register(name="Group B")
    group3 = groups.register(name="Group C")

    # Correct
    std_user_id = users.query(email="stduser@gmail.com")[0].id
    admin_user = users.query(email="admin_user@gmail.com")[0].id
    msg_content = {
        "user_id": std_user_id,
        "groups": [group1.id],
        "current_user": admin_user,
    }
    assert (
        len(
            database.session.query(UserGroup)
            .filter_by(user=std_user_id, group=group1.id)
            .all()
        )
        == 0
    )
    response = build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
    assert response.status_code == 200
    assert response.content == {"message": "User updated successfully!"}
    assert (
        len(
            database.session.query(UserGroup)
            .filter_by(user=std_user_id, group=group1.id)
            .all()
        )
        == 1
    )
    assert (
        database.session.query(UserGroup)
        .filter_by(user=std_user_id, group=group1.id)
        .first()
        .id
        == group1.id
    )

    msg_content = {
        "user_id": std_user_id,
        "groups": [group2.id, group3.id],
        "current_user": admin_user,
    }
    response = build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
    assert response.status_code == 200
    assert response.content == {"message": "User updated successfully!"}
    assert (
        len(
            database.session.query(UserGroup)
            .filter_by(user=std_user_id, group=group1.id)
            .all()
        )
        == 0
    )
    assert len(database.session.query(UserGroup).filter_by(user=std_user_id).all()) == 2
    assert (
        len(
            database.session.query(UserGroup)
            .filter_by(user=std_user_id, group=group2.id)
            .all()
        )
        == 1
    )
    assert (
        len(
            database.session.query(UserGroup)
            .filter_by(user=std_user_id, group=group3.id)
            .all()
        )
        == 1
    )

    msg_content = {
        "user_id": admin_user,
        "groups": [group2.id],
        "current_user": admin_user,
    }
    response = build_syft_msg(domain, UpdateUserMessage, msg_content, generic_key)
    assert response.status_code == 200
    assert response.content == {"message": "User updated successfully!"}
    assert len(database.session.query(UserGroup).filter_by(group=group2.id).all()) == 2
