import jwt
import pytest
from bcrypt import checkpw
from flask import current_app as app
from src.app.main.core.exceptions import PyGridError
from src.app.main.database import *
from src.app.main.events.user_related import *

JSON_DECODE_ERR_MSG = (
    "Expecting property name enclosed in " "double quotes: line 1 column 2 (char 1)"
)
owner_role = ("Owner", True, True, True, True, True, True, True)
user_role = ("User", False, False, False, False, False, False, False)
admin_role = ("Administrator", True, True, True, True, False, False, True)
user1 = (
    "tech@gibberish.com",
    "BDEB6E8EE39B6C70835993486C9E65DC",
    "]GBF[R>GX[9Cmk@DthFT!mhloUc%[f",
    "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    1,
)
user2 = (
    "tech@gibberish.com",
    "2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
    "$2b$12$tufn64/0gSIAdprqBrRzC.",
    "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    1,
)
user3 = (
    "anemail@anemail.com",
    "2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
    "$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
    "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
    2,
)

user4 = (
    "tech@gibberish.com",
    "2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
    "$2b$12$tufn64/0gSIAdprqBrRzC.",
    "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    2,
)
user5 = (
    "owner@owner.com",
    "RcEEa25p/APCVGFaBaiZpytLieFsv22",
    "$2b$12$OazL5oj8/lxxOV5a5j2Nme",
    "4de2d41486ceaffdf0c1778e50cea00000d6549ffe808fa860ecd4e91d9ee1b1",
    1,
)


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


def test_post_first_user_success(database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_role = create_role(*user_role)
    database.session.add(new_role)

    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    message = {"email": "someemail@email.com", "password": "123secretpassword"}
    result = signup_user_socket(message)
    assert result["success"] == True
    assert result["user"]["id"] == 2
    assert len(result["user"]["private_key"]) == 64
    assert result["user"]["role"]["id"] == 2
    assert result["user"]["email"] == "someemail@email.com"


def test_post_first_user_missing_role(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    message = {"email": "someemail@email.com", "password": "123secretpassword"}
    result = signup_user_socket(message)
    assert result["error"] == "Role ID not found!"


def test_post_user_with_role(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "email": "someemail@email.com",
        "password": "123secretpassword",
        "role": 1,
    }
    result = signup_user_socket(message)
    assert result["success"] == True
    assert result["user"]["id"] == 2
    assert len(result["user"]["private_key"]) == 64
    assert result["user"]["role"]["id"] == 1
    assert result["user"]["email"] == "someemail@email.com"


def test_post_user_invalid_key(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "private-key": "alasthiskeyisntvalid",
        "email": "someemail@email.com",
        "password": "123secretpassword",
        "role": 1,
    }
    result = signup_user_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_post_user_with_missing_role(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "email": "someemail@email.com",
        "password": "123secretpassword",
        "role": 3,
    }
    result = signup_user_socket(message)
    assert result["error"] == "Role ID not found!"


def test_login_user_valid_credentials(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "email": "tech@gibberish.com",
        "password": "&UP!SN!;J4Mx;+A]",
    }
    result = login_user_socket(message)
    assert result["success"] == True
    token = result["token"]
    content = jwt.decode(token, app.config["SECRET_KEY"], algorithms="HS256")
    assert content["id"] == 1


def test_login_user_invalid_key(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "private-key": "imaninvalidkeyalright",
        "email": "tech@gibberish.com",
        "password": "&UP!SN!;J4Mx;+A]",
    }
    result = login_user_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_login_user_missing_key(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    message = {"email": "tech@gibberish.com", "password": "&UP!SN!;J4Mx;+A]"}
    result = login_user_socket(message)
    assert result["error"] == "Missing request key!"


def test_login_user_invalid_email(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "email": "perhaps@perhaps.com",
        "password": "&UP!SN!;J4Mx;+A]",
    }
    result = login_user_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_login_user_invalid_password(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "email": "tech@gibberish.com",
        "password": "@123456notmypassword",
    }
    result = login_user_socket(message)
    assert result["error"] == "Invalid credentials!"


# GET ALL USERS


def test_get_users_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = get_all_users_socket(message)
    assert len(result["users"]) == 2
    assert result["users"][0]["id"] == 1
    assert result["users"][1]["id"] == 2


def test_get_users_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
    }
    result = get_all_users_socket(message)
    assert result["error"] == "User is not authorized for this operation!"


def test_get_users_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {"token": token.decode("UTF-8")}
    result = get_all_users_socket(message)
    assert result["error"] == "Missing request key!"


def test_get_users_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced"
    }
    result = get_all_users_socket(message)
    assert result["error"] == "Missing request key!"


def test_get_users_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "invalid312987as12they0come",
        "token": token.decode("UTF-8"),
    }
    result = get_all_users_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_get_users_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "peppperplsiwouldhavesome")
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = get_all_users_socket(message)
    assert result["error"] == "Invalid credentials!"


# GET SPECIFIC USER


def test_get_one_user_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "id": 2,
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = get_specific_user_socket(message)
    assert result["user"]["id"] == 2
    assert result["user"]["email"] == "anemail@anemail.com"


def test_get_one_user_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {"id": 1, "token": token.decode("UTF-8")}
    result = get_specific_user_socket(message)
    assert result["error"] == "Missing request key!"


def test_get_one_user_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "id": 1,
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    }
    result = get_specific_user_socket(message)
    assert result["error"] == "Missing request key!"


def test_get_one_user_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "id": 1,
        "private-key": "invalid312987as12they0come",
        "token": token.decode("UTF-8"),
    }
    result = get_specific_user_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_get_one_user_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "peppperplsiwouldhavesome")
    message = {
        "id": 2,
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = get_specific_user_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_get_one_user_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "id": 1,
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
    }
    result = get_specific_user_socket(message)
    assert result["error"] == "User is not authorized for this operation!"


def test_get_one_missing_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "id": 3,
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = get_specific_user_socket(message)
    assert result["error"] == "User ID not found!"


# PUT USER EMAIL


def test_put_other_user_email_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "id": 2,
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "email": "brandnew@brandnewemail.com",
    }
    result = change_user_email_socket(message)
    assert result["user"]["id"] == 2
    assert result["user"]["email"] == "brandnew@brandnewemail.com"
    assert database.session.query(User).get(2).email == "brandnew@brandnewemail.com"


def test_put_other_user_email_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "id": 2,
        "token": token.decode("UTF-8"),
        "email": "brandnew@brandnewemail.com",
    }
    result = change_user_email_socket(message)
    assert result["error"] == "Missing request key!"


def test_put_other_user_email_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced"
    }
    message = {"id": 2, "email": "brandnew@brandnewemail.com"}
    result = change_user_email_socket(message)
    assert result["error"] == "Missing request key!"


def test_put_user_email_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "id": 2,
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "email": "brandnew@brandnewemail.com",
    }
    result = change_user_email_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_put_user_email_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    token = jwt.encode({"id": 1}, "secretitis")
    message = {
        "id": 2,
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "email": "brandnew@brandnewemail.com",
    }
    result = change_user_email_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_put_other_user_email_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "id": 1,
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "email": "brandnew@brandnewemail.com",
    }
    result = change_user_email_socket(message)
    assert result["error"] == "User is not authorized for this operation!"


def test_put_own_user_email_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "id": 2,
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "email": "brandnew@brandnewemail.com",
    }
    result = change_user_email_socket(message)
    assert result["user"]["id"] == 2
    assert result["user"]["email"] == "brandnew@brandnewemail.com"
    assert database.session.query(User).get(2).email == "brandnew@brandnewemail.com"


def test_put_user_email_missing_role(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "id": 2,
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "email": "brandnew@brandnewemail.com",
    }
    result = change_user_email_socket(message)
    assert result["error"] == "Role ID not found!"


def test_put_other_user_email_missing_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "id": 2,
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "email": "brandnew@brandnewemail.com",
    }
    result = change_user_email_socket(message)
    assert result["error"] == "User ID not found!"


# PUT USER ROLE


def test_put_other_user_role_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).role == 2

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "role": 1,
        "id": 2,
    }
    result = change_user_role_socket(message)
    assert result["user"]["id"] == 2
    assert result["user"]["role"]["id"] == 1
    assert database.session.query(User).get(2).role == 1


def test_put_other_user_role_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {"token": token.decode("UTF-8"), "role": 1, "id": 2}
    result = change_user_role_socket(message)
    assert result["error"] == "Missing request key!"


def test_put_other_user_role_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced"
    }
    message = {"role": 1, "id": 2}
    result = change_user_role_socket(message)
    assert result["error"] == "Missing request key!"


def test_put_user_role_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "role": 1,
        "id": 2,
    }
    result = change_user_role_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_put_user_role_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "secretitis")
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "role": 1,
        "id": 2,
    }
    result = change_user_role_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_put_other_user_role_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "role": 2,
        "id": 1,
    }
    result = change_user_role_socket(message)
    assert result["error"] == "User is not authorized for this operation!"


def test_put_own_user_role_sucess(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user5)
    database.session.add(new_user)
    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=3,
    )
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).role == 2

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "role": 3,
        "id": 2,
    }
    result = change_user_role_socket(message)
    assert result["user"]["id"] == 2
    assert result["user"]["role"]["id"] == 3
    assert database.session.query(User).get(2).role == 3


def test_put_first_user_unauthorized(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user5)
    database.session.add(new_user)
    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=3,
    )
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "role": 3,
        "id": 1,
    }
    result = change_user_role_socket(message)
    assert result["error"] == "User is not authorized for this operation!"


def test_put_other_user_role_owner_unauthorized(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user5)
    database.session.add(new_user)
    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=3,
    )
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "role": 1,
        "id": 3,
    }
    result = change_user_role_socket(message)
    assert result["error"] == "User is not authorized for this operation!"


def test_put_other_user_role_owner_success(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user5)
    database.session.add(new_user)
    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=3,
    )
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(3).role == 3

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "4de2d41486ceaffdf0c1778e50cea00000d6549ffe808fa860ecd4e91d9ee1b1",
        "token": token.decode("UTF-8"),
        "role": 1,
        "id": 3,
    }
    result = change_user_role_socket(message)
    assert result["user"]["id"] == 3
    assert result["user"]["role"]["id"] == 1
    assert database.session.query(User).get(3).role == 1


def test_put_user_role_missing_role(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "role": 2,
        "id": 2,
    }
    result = change_user_role_socket(message)
    assert result["error"] == "Role ID not found!"


def test_put_other_user_role_missing_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "role": 2,
        "id": 2,
    }
    result = change_user_role_socket(message)
    assert result["error"] == "User ID not found!"


# PUT USER PASSWORD


def test_put_other_user_password_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="wi6hJCTz9QN1GcKc2ZJk7ReZ1LshNsu",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=2,
    )
    database.session.add(new_user)

    database.session.commit()

    user = database.session.query(User).get(2)
    assert checkpw(
        b"ownerpassword123@@",
        user.salt.encode("UTF-8") + user.hashed_password.encode("UTF-8"),
    )

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    new_password = "BrandNewPassword123"
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "password": new_password,
        "id": 2,
    }

    result = change_user_password_socket(message)
    assert result["user"]["id"] == 2
    assert checkpw(
        new_password.encode("UTF-8"),
        user.salt.encode("UTF-8") + user.hashed_password.encode("UTF-8"),
    )


def test_put_user_password_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    new_password = "BrandNewPassword123"
    message = {"token": token.decode("UTF-8"), "id": 2, "password": new_password}

    result = change_user_password_socket(message)
    assert result["error"] == "Missing request key!"


def test_put_user_password_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    new_password = "BrandNewPassword123"
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "id": 2,
        "password": new_password,
    }
    result = change_user_password_socket(message)
    assert result["error"] == "Missing request key!"


def test_put_user_password_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    new_password = "BrandNewPassword123"
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "id": 2,
        "password": new_password,
    }
    result = change_user_password_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_put_user_password_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "secretitis")
    new_password = "BrandNewPassword123"
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 2,
        "password": new_password,
    }
    result = change_user_password_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_put_other_user_password_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    new_password = "BrandNewPassword123"
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "id": 1,
        "password": new_password,
    }
    result = change_user_password_socket(message)
    assert result["error"] == "User is not authorized for this operation!"


def test_put_own_user_password_success(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user5)
    database.session.add(new_user)
    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="wi6hJCTz9QN1GcKc2ZJk7ReZ1LshNsu",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=3,
    )
    database.session.add(new_user)

    database.session.commit()

    user = database.session.query(User).get(3)
    assert checkpw(
        b"ownerpassword123@@",
        user.salt.encode("UTF-8") + user.hashed_password.encode("UTF-8"),
    )

    token = jwt.encode({"id": 3}, app.config["SECRET_KEY"])
    new_password = "BrandNewPassword123"
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "id": 3,
        "password": new_password,
    }
    result = change_user_password_socket(message)
    assert result["user"]["id"] == 3
    assert checkpw(
        new_password.encode("UTF-8"),
        user.salt.encode("UTF-8") + user.hashed_password.encode("UTF-8"),
    )


def test_put_other_user_email_missing_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    new_password = "BrandNewPassword123"
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 2,
        "password": new_password,
    }
    result = change_user_password_socket(message)
    assert result["error"] == "User ID not found!"


# PUT USER GROUPS


def test_put_other_user_groups_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="wi6hJCTz9QN1GcKc2ZJk7ReZ1LshNsu",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=2,
    )
    database.session.add(new_user)
    new_group = Group(name="Hospital_X")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Y")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Z")
    database.session.add(new_group)
    new_usergroup = UserGroup(user=2, group=1)
    database.session.add(new_usergroup)

    database.session.commit()

    user_groups = database.session.query(UserGroup).filter_by(user=2).all()
    assert len(user_groups) == 1
    assert user_groups[0].group == 1

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 2,
        "groups": [2, 3],
    }
    result = change_user_groups_socket(message)
    user_groups = database.session.query(UserGroup).filter_by(user=2).all()

    assert result["user"]["id"] == 2
    assert len(result["user"]["groups"]) == 2
    assert result["user"]["groups"][0]["id"] == 2
    assert result["user"]["groups"][1]["id"] == 3

    assert len(user_groups) == 2
    assert user_groups[0].group == 2
    assert user_groups[1].group == 3


def test_put_user_groups_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)
    database.session.add(new_user)
    new_group = Group(name="Hospital_X")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Y")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Z")
    database.session.add(new_group)
    new_usergroup = UserGroup(user=2, group=1)
    database.session.add(new_usergroup)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {"token": token.decode("UTF-8"), "id": 2, "groups": [2, 3]}
    result = change_user_groups_socket(message)
    user_groups = database.session.query(UserGroup).filter_by(user=2).all()

    assert result["error"] == "Missing request key!"


def test_put_user_groups_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)
    new_group = Group(name="Hospital_X")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Y")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Z")
    database.session.add(new_group)
    new_usergroup = UserGroup(user=2, group=1)
    database.session.add(new_usergroup)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "groups": [2, 3],
        "id": 2,
    }
    result = change_user_groups_socket(message)
    assert result["error"] == "Missing request key!"


def test_put_user_groups_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)
    new_group = Group(name="Hospital_X")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Y")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Z")
    database.session.add(new_group)
    new_usergroup = UserGroup(user=2, group=1)
    database.session.add(new_usergroup)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "id": 2,
        "groups": [2, 3],
    }
    result = change_user_groups_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_put_user_groups_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)
    new_group = Group(name="Hospital_X")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Y")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Z")
    database.session.add(new_group)
    new_usergroup = UserGroup(user=2, group=1)
    database.session.add(new_usergroup)

    database.session.commit()

    token = jwt.encode({"id": 1}, "secretitis")
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 2,
        "groups": [2, 3],
    }
    result = change_user_groups_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_put_other_user_groups_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)
    new_group = Group(name="Hospital_X")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Y")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Z")
    database.session.add(new_group)
    new_usergroup = UserGroup(user=2, group=1)
    database.session.add(new_usergroup)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "id": 1,
        "groups": [2, 3],
    }
    result = change_user_groups_socket(message)
    assert result["error"] == "User is not authorized for this operation!"


def test_put_own_user_groups_success(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user5)
    database.session.add(new_user)
    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="wi6hJCTz9QN1GcKc2ZJk7ReZ1LshNsu",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=3,
    )
    database.session.add(new_user)
    new_group = Group(name="Hospital_X")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Y")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Z")
    database.session.add(new_group)
    new_usergroup = UserGroup(user=2, group=1)
    database.session.add(new_usergroup)
    new_usergroup = UserGroup(user=3, group=2)
    database.session.add(new_usergroup)

    database.session.commit()

    user_groups = database.session.query(UserGroup).filter_by(user=3).all()
    assert len(user_groups) == 1
    assert user_groups[0].group == 2

    token = jwt.encode({"id": 3}, app.config["SECRET_KEY"])
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "id": 3,
        "groups": [1],
    }
    result = change_user_groups_socket(message)
    user_groups = database.session.query(UserGroup).filter_by(user=3).all()

    assert result["user"]["id"] == 3
    assert len(result["user"]["groups"]) == 1
    assert result["user"]["groups"][0]["id"] == 1

    assert len(user_groups) == 1
    assert user_groups[0].group == 1


def test_put_other_user_groups_missing_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_group = Group(name="Hospital_X")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Y")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Z")
    database.session.add(new_group)
    new_usergroup = UserGroup(user=2, group=1)
    database.session.add(new_usergroup)
    new_usergroup = UserGroup(user=3, group=2)
    database.session.add(new_usergroup)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "id": 2,
        "groups": [1],
    }
    result = change_user_groups_socket(message)
    assert result["error"] == "User ID not found!"


def test_put_user_groups_missing_group(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user5)
    database.session.add(new_user)
    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="wi6hJCTz9QN1GcKc2ZJk7ReZ1LshNsu",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=3,
    )
    database.session.add(new_user)
    new_group = Group(name="Hospital_X")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Y")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Z")
    database.session.add(new_group)
    new_usergroup = UserGroup(user=2, group=1)
    database.session.add(new_usergroup)
    new_usergroup = UserGroup(user=3, group=2)
    database.session.add(new_usergroup)

    database.session.commit()

    token = jwt.encode({"id": 3}, app.config["SECRET_KEY"])
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "id": 3,
        "groups": [5],
    }
    result = change_user_groups_socket(message)
    user_groups = database.session.query(UserGroup).filter_by(user=3).all()

    assert result["error"] == "Group ID not found!"


# DELETE USER


def test_delete_other_user_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="wi6hJCTz9QN1GcKc2ZJk7ReZ1LshNsu",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=2,
    )
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2) is not None

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "id": 2,
        "token": token.decode("UTF-8"),
    }
    result = delete_user_socket(message)
    assert database.session.query(User).get(2) is None


def test_delete_user_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {"id": 1, "token": token.decode("UTF-8")}
    result = delete_user_socket(message)
    assert result["error"] == "Missing request key!"


def test_delete_user_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "id": 2,
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    }
    result = delete_user_socket(message)
    assert result["error"] == "Missing request key!"


def test_delete_user_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "id": 2,
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
    }
    result = delete_user_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_delete_user_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "secretitis")
    message = {
        "id": 2,
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = delete_user_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_delete_other_user_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "id": 1,
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
    }
    result = delete_user_socket(message)
    assert result["error"] == "User is not authorized for this operation!"


def test_delete_own_user_success(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user5)
    database.session.add(new_user)
    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="wi6hJCTz9QN1GcKc2ZJk7ReZ1LshNsu",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=3,
    )
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(3) is not None

    token = jwt.encode({"id": 3}, app.config["SECRET_KEY"])
    message = {
        "id": 3,
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
    }
    result = delete_user_socket(message)
    user_groups = database.session.query(UserGroup).filter_by(user=3).all()

    assert database.session.query(User).get(3) is None


def test_delete_other_user_missing_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "id": 2,
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
    }
    result = delete_user_socket(message)
    assert result["error"] == "User ID not found!"


# SEARCH USERS


def test_search_users_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="wi6hJCTz9QN1GcKc2ZJk7ReZ1LshNsu",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=2,
    )
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "email": "anemail@anemail.com",
    }
    result = search_users_socket(message)
    assert result["success"] == True
    assert len(result["users"]) == 1
    assert result["users"][0]["id"] == 2


def test_search_users_nomatch(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user5)
    database.session.add(new_user)
    new_user = User(
        email="tech@gibberish.com",
        hashed_password="2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
        salt="$2b$12$tufn64/0gSIAdprqBrRzC.",
        private_key="fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        role=3,
    )
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="wi6hJCTz9QN1GcKc2ZJk7ReZ1LshNsu",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=3,
    )
    database.session.add(new_user)

    new_group = Group(name="Hospital_X")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Y")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Z")
    database.session.add(new_group)
    new_usergroup = UserGroup(user=1, group=3)
    database.session.add(new_usergroup)
    new_usergroup = UserGroup(user=2, group=1)
    database.session.add(new_usergroup)
    new_usergroup = UserGroup(user=3, group=1)
    database.session.add(new_usergroup)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "4de2d41486ceaffdf0c1778e50cea00000d6549ffe808fa860ecd4e91d9ee1b1",
        "token": token.decode("UTF-8"),
        "role": 3,
        "group": 3,
    }
    result = search_users_socket(message)
    assert result["success"] == True
    assert len(result["users"]) == 0


def test_search_users_onematch(client, database, cleanup):
    new_role = new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = User(
        email="tech@gibberish.com",
        hashed_password="RcEEa25p/APCVGFaBaiZpytLieFsv22",
        salt="$2b$12$OazL5oj8/lxxOV5a5j2Nme",
        private_key="4de2d41486ceaffdf0c1778e50cea00000d6549ffe808fa860ecd4e91d9ee1b1",
        role=1,
    )
    database.session.add(new_user)
    new_user = User(
        email="tech@gibberish.com",
        hashed_password="2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
        salt="$2b$12$tufn64/0gSIAdprqBrRzC.",
        private_key="fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        role=3,
    )
    database.session.add(new_user)
    new_user = User(
        email="anemail@anemail.com",
        hashed_password="wi6hJCTz9QN1GcKc2ZJk7ReZ1LshNsu",
        salt="$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
        private_key="acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        role=3,
    )
    database.session.add(new_user)

    new_group = Group(name="Hospital_X")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Y")
    database.session.add(new_group)
    new_group = Group(name="Hospital_Z")
    database.session.add(new_group)
    new_usergroup = UserGroup(user=1, group=3)
    database.session.add(new_usergroup)
    new_usergroup = UserGroup(user=2, group=1)
    database.session.add(new_usergroup)
    new_usergroup = UserGroup(user=3, group=1)
    database.session.add(new_usergroup)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "4de2d41486ceaffdf0c1778e50cea00000d6549ffe808fa860ecd4e91d9ee1b1",
        "token": token.decode("UTF-8"),
        "role": 3,
        "group": 1,
        "email": "tech@gibberish.com",
    }
    result = search_users_socket(message)
    assert result["success"] == True
    assert len(result["users"]) == 1
    assert result["users"][0]["id"] == 2


def test_search_users_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {"token": token.decode("UTF-8"), "email": "anemail@anemail.com"}
    result = search_users_socket(message)
    assert result["error"] == "Missing request key!"


def test_search_users_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "email": "anemail@anemail.com",
    }
    result = search_users_socket(message)
    assert result["error"] == "Missing request key!"


def test_search_users_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "email": "anemail@anemail.com",
    }
    result = search_users_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_search_users_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "secretitis")
    message = {
        "private-key": "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        "token": token.decode("UTF-8"),
        "email": "anemail@anemail.com",
    }
    result = search_users_socket(message)
    assert result["error"] == "Invalid credentials!"


def test_search_users_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user2)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    message = {
        "private-key": "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
        "token": token.decode("UTF-8"),
        "email": "anemail@anemail.com",
    }
    result = search_users_socket(message)
    assert result["error"] == "User is not authorized for this operation!"
