# third party
from bcrypt import checkpw
import pytest
from src.main.core.database import *
from src.main.core.exceptions import InvalidCredentialsError
from src.main.core.manager import UserManager

user_role = ("User", False, False, False, False, False, False, False)
admin_role = ("Administrator", True, True, True, True, False, False, True)


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


def test_create_user_manager(database, cleanup):
    users = UserManager(database)


def test_register_new_user(database, cleanup):
    users = UserManager(database)
    test_role = create_role(*user_role)
    user = users.register(
        email="testing@email.com",
        hashed_password="aifhufhaf",
        salt="aifhaufa",
        private_key="afihauhfao",
        role=test_role.id,
    )

    assert user.email == "testing@email.com"
    assert user.hashed_password == "aifhufhaf"
    assert user.salt == "aifhaufa"
    assert user.private_key == "afihauhfao"
    assert user.role == test_role.id


def test_query_new_user(database, cleanup):
    users = UserManager(database)

    test_role = create_role(*user_role)
    user1 = {
        "email": "user1@email.com",
        "hashed_password": "afhuefhaare",
        "salt": "diwriqjroqds",
        "private_key": "rweqoasnfa",
        "role": test_role.id,
    }

    user2 = {
        "email": "user2@email.com",
        "hashed_password": "rerfsfa",
        "salt": "dgfgsgwrwers",
        "private_key": "AHEIJASDAEW",
        "role": test_role.id,
    }

    db_user1 = users.register(**user1)
    db_user2 = users.register(**user2)

    # Test every database field, except role id
    for key, value in list(user1.items())[:-1]:
        query_result_1 = users.query(**{key: value})

        assert len(query_result_1) == 1
        query_result_1 = query_result_1[0]
        assert query_result_1.email == db_user1.email
        assert query_result_1.hashed_password == db_user1.hashed_password
        assert query_result_1.salt == db_user1.salt
        assert query_result_1.private_key == db_user1.private_key
        assert query_result_1.role == db_user1.role

    for key, value in list(user2.items())[:-1]:
        query_result_2 = users.query(**{key: value})

        assert len(query_result_2) == 1
        query_result_2 = query_result_2[0]
        assert query_result_2.email == db_user2.email
        assert query_result_2.hashed_password == db_user2.hashed_password
        assert query_result_2.salt == db_user2.salt
        assert query_result_2.private_key == db_user2.private_key
        assert query_result_2.role == db_user2.role

    query_result_3 = users.query(role=test_role.id)
    assert len(query_result_3) == 2


def test_set_email(database, cleanup):
    users = UserManager(database)

    test_role = create_role(*user_role)
    user1 = {
        "email": "user1@email.com",
        "hashed_password": "afhuefhaare",
        "salt": "diwriqjroqds",
        "private_key": "rweqoasnfa",
        "role": test_role.id,
    }

    db_user1 = users.register(**user1)

    assert users.query(id=db_user1.id)[0].email == "user1@email.com"

    users.set(user_id=db_user1.id, email="newemail@email.com")

    assert users.query(id=db_user1.id)[0].email == "newemail@email.com"


def test_set_password(database, cleanup):
    users = UserManager(database)

    test_role = create_role(*user_role)
    user1 = {
        "email": "user1@email.com",
        "hashed_password": "afhuefhaare",
        "salt": "diwriqjroqds",
        "private_key": "rweqoasnfa",
        "role": test_role.id,
    }

    db_user1 = users.register(**user1)

    assert users.query(id=db_user1.id)[0].hashed_password == "afhuefhaare"

    users.set(user_id=db_user1.id, password="new_password")

    assert users.login(email="user1@email.com", password="new_password")


def test_set_role(database, cleanup):
    users = UserManager(database)

    user_role_obj = create_role(*user_role)
    admin_role_obj = create_role(*admin_role)

    user1 = {
        "email": "user1@email.com",
        "hashed_password": "afhuefhaare",
        "salt": "diwriqjroqds",
        "private_key": "rweqoasnfa",
        "role": user_role_obj.id,
    }

    db_user1 = users.register(**user1)

    assert users.query(id=db_user1.id)[0].role == user_role_obj.id

    users.set(user_id=db_user1.id, role=admin_role_obj.id)

    assert users.query(id=db_user1.id)[0].role == admin_role_obj.id


def test_signup(database, cleanup):
    users = UserManager(database)

    user_role_obj = create_role(*user_role)

    users.signup(
        email="testing@email.com",
        password="qrjhsiofjadasd",
        role=user_role_obj.id,
        private_key="aghuehffadawe",
        verify_key="aufhyfaeiiead",
    )

    user = users.query(email="testing@email.com")[0]

    assert user.email == "testing@email.com"
    assert user.role == user_role_obj.id
    assert checkpw(
        "qrjhsiofjadasd".encode("UTF-8"),
        user.salt.encode("UTF-8") + user.hashed_password.encode("UTF-8"),
    )


def test_login(database, cleanup):
    users = UserManager(database)

    user_role_obj = create_role(*user_role)

    users.signup(
        email="testing@email.com",
        password="qrjhsiofjadasd",
        role=user_role_obj.id,
        private_key="aghuehffadawe",
        verify_key="aiehufaefhuada",
    )

    # Success
    user = users.login(email="testing@email.com", password="qrjhsiofjadasd")

    # Wrong e-mail
    with pytest.raises(InvalidCredentialsError) as exc:
        users.login(email="wrongemail@email.com", password="qrjhsiofjadasd")

    # Wrong password
    with pytest.raises(InvalidCredentialsError) as exc:
        users.login(email="testing@email.com", password="qrjhsiofja")
