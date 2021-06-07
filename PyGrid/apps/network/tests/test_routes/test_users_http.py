# stdlib
from json import dumps
from json import loads

# third party
from bcrypt import checkpw
from flask import current_app as app
import jwt
import pytest
from src.main.core.database import *

JSON_DECODE_ERR_MSG = (
    "Expecting property name enclosed in " "double quotes: line 1 column 2 (char 1)"
)
owner_role = ("Owner", True, True, True, True)
user_role = ("User", False, False, False, False)
admin_role = ("Administrator", True, True, True, False)

user1 = (
    "tech@gibberish.com",
    "BDEB6E8EE39B6C70835993486C9E65DC",
    "]GBF[R>GX[9Cmk@DthFT!mhloUc%[f",
    "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    1,
)
user2 = (
    "anemail@anemail.com",
    "2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
    "$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
    "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
    2,
)
user3 = (
    "anemail@anemail.com",
    "2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
    "$2b$12$rj8MnLcKBxAgL7GUHrYn6O",
    "acfc10d15d7ec9f7cd05a312489af2794619c6f11e9af34671a5f33da48c1de2",
    3,
)
user4 = (
    "tech@gibberish.com",
    "2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
    "$2b$12$tufn64/0gSIAdprqBrRzC.",
    "fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
    2,
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


# POST USER


def test_post_role_user_data_no_key(client):
    result = client.post("/users", data="{bad", content_type="application/json")
    assert result.status_code == 400


def test_post_user_bad_data_with_key(client, database, cleanup):
    new_role = create_role(*owner_role)
    new_user = create_user(*user1)
    database.session.add(new_role)
    database.session.add(new_user)
    database.session.commit()

    result = client.post("/users", data="{bad", content_type="application/json")

    assert result.status_code == 400


def test_post_std_user_success(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    payload = {"email": "someemail@email.com", "password": "123secretpassword"}
    result = client.post("/users", data=dumps(payload), content_type="application/json")

    assert result.status_code == 200
    assert result.get_json() == {"message": "User created successfully!"}


def test_post_std_user_missing_role(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    payload = {
        "email": "someemail@email.com",
        "password": "123secretpassword",
        "role": "Unexpected Role",
    }
    result = client.post(
        "/users", data=dumps(payload), headers=headers, content_type="application/json"
    )

    assert result.get_json() == {"error": "Role ID not found!"}


# TODO: Update unit tests below
def test_post_user_with_role(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    headers = {}

    payload = {
        "email": "someemail@email.com",
        "password": "123secretpassword",
        "role": "User",
    }

    result = client.post(
        "/users", data=dumps(payload), headers=headers, content_type="application/json"
    )

    assert result.status_code == 200
    assert result.get_json() == {"message": "User created successfully!"}


def test_login_user_valid_credentials(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = User(
        email="tech@gibberish.com",
        hashed_password="2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
        salt="$2b$12$tufn64/0gSIAdprqBrRzC.",
        private_key="fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        role=1,
    )
    database.session.add(new_user)

    database.session.commit()

    headers = {}
    payload = {"email": "tech@gibberish.com", "password": "&UP!SN!;J4Mx;+A]"}
    result = client.post(
        "/users/login",
        data=dumps(payload),
        headers=headers,
        content_type="application/json",
    )

    assert result.status_code == 200

    token = result.get_json()["token"]
    content = jwt.decode(token, app.config["SECRET_KEY"], algorithms="HS256")
    assert content["id"] == 1


def test_login_user_invalid_email(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    payload = {"email": "p@perhaps.com", "password": "&UP!SN!;J4Mx;+A]"}
    result = client.post(
        "/users/login",
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_login_user_invalid_password(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = User(
        email="tech@gibberish.com",
        hashed_password="2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
        salt="$2b$12$tufn64/0gSIAdprqBrRzC.",
        private_key="fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        role=1,
    )
    database.session.add(new_user)

    database.session.commit()

    headers = {}
    payload = {"email": "tech@gibberish.com", "password": "@123456notmypassword"}
    result = client.post(
        "/users/login",
        data=dumps(payload),
        headers=headers,
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


# GET ALL USERS


def test_get_users_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get("/users", headers=headers, content_type="application/json")

    assert result.status_code == 200
    assert len(result.get_json()) == 2
    assert result.get_json()[0]["id"] == 1
    assert result.get_json()[1]["id"] == 2


def test_get_users_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    headers = {}
    result = client.get("/users", headers=headers, content_type="application/json")

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_get_users_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "peppperplsiwouldhavesome")
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get("/users", headers=headers, content_type="application/json")

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


# GET SPECIFIC USER


def test_get_one_user_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get("/users/2", headers=headers, content_type="application/json")

    assert result.status_code == 200
    assert result.get_json()["id"] == 2
    assert result.get_json()["email"] == "anemail@anemail.com"


def test_get_one_user_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    headers = {}
    result = client.get("/users/1", headers=headers, content_type="application/json")

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_get_one_user_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "peppperplsiwouldhavesome")
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get("/users/2", headers=headers, content_type="application/json")

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_get_one_missing_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get("/users/3", headers=headers, content_type="application/json")

    assert result.status_code == 404
    assert result.get_json()["error"] == "User not found!"


# PUT USER EMAIL


def test_put_other_user_email_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"email": "brandnew@brandnewemail.com"}
    result = client.put(
        "/users/2/email",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 200
    assert result.get_json() == {"message": "User updated successfully!"}


def test_put_other_user_email_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    headers = {}
    payload = {"email": "brandnew@brandnewemail.com"}
    result = client.put(
        "/users/2/email",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_put_user_email_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    token = jwt.encode({"id": 1}, "secretitis")
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"email": "brandnew@brandnewemail.com"}
    result = client.put(
        "/users/2/email",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_put_other_user_email_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"email": "brandnew@brandnewemail.com"}
    result = client.put(
        "/users/1/email",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "You're not allowed to change other user data!"


def test_put_own_user_email_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"email": "brandnew@brandnewemail.com"}
    result = client.put(
        "/users/2/email",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 200
    assert result.get_json() == {"message": "User updated successfully!"}


def test_put_user_email_missing_role(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).email == "anemail@anemail.com"

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"email": "brandnew@brandnewemail.com"}
    result = client.put(
        "/users/2/email",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 200
    assert result.get_json() == {"message": "User updated successfully!"}


def test_put_other_user_email_missing_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"email": "brandnew@brandnewemail.com"}
    result = client.put(
        "/users/5/email",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 404
    assert result.get_json()["error"] == "User not found!"


# PUT USER ROLE


def test_put_other_user_role_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).role == 2

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"role": 1}
    result = client.put(
        "/users/2/role",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 200
    assert result.get_json() == {"message": "User updated successfully!"}


def test_put_other_user_role_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    headers = {}
    payload = {"role": 1}
    result = client.put(
        "/users/2/role",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_put_user_role_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "secretitis")
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"role": 1}
    result = client.put(
        "/users/2/role",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_put_other_user_role_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"role": 2}
    result = client.put(
        "/users/1/role",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "You're not allowed to change other user data!"


def test_put_own_user_role_sucess(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = User(
        email="owner@owner.com",
        hashed_password="RcEEa25p/APCVGFaBaiZpytLieFsv22",
        salt="$2b$12$OazL5oj8/lxxOV5a5j2Nme",
        private_key="4de2d41486ceaffdf0c1778e50cea00000d6549ffe808fa860ecd4e91d9ee1b1",
        role=1,
    )
    database.session.add(new_user)
    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(2).role == 2

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"role": 3}
    result = client.put(
        "/users/2/role",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 200
    assert result.get_json() == {"message": "User updated successfully!"}


def test_put_first_user_unauthorized(client, database, cleanup):
    # Create Owner role
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    # Create Admin role
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    # Create User role
    new_role = create_role(*user_role)
    database.session.add(new_role)

    # New Owner User
    new_user = User(
        email="owner@owner.com",
        hashed_password="RcEEa25p/APCVGFaBaiZpytLieFsv22",
        salt="$2b$12$OazL5oj8/lxxOV5a5j2Nme",
        private_key="4de2d41486ceaffdf0c1778e50cea00000d6549ffe808fa860ecd4e91d9ee1b1",
        role=1,
    )
    database.session.add(new_user)

    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"role": 3}
    result = client.put(
        "/users/1/role",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 403
    assert (
        result.get_json()["error"] == "You're not allowed to change Owner user roles!"
    )


def test_put_other_user_role_owner_unauthorized(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = User(
        email="owner@owner.com",
        hashed_password="RcEEa25p/APCVGFaBaiZpytLieFsv22",
        salt="$2b$12$OazL5oj8/lxxOV5a5j2Nme",
        private_key="4de2d41486ceaffdf0c1778e50cea00000d6549ffe808fa860ecd4e91d9ee1b1",
        role=1,
    )
    database.session.add(new_user)
    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"role": 1}
    result = client.put(
        "/users/3/role",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "You can't change it to Owner role!"


def test_put_other_user_role_owner(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = User(
        email="owner@owner.com",
        hashed_password="RcEEa25p/APCVGFaBaiZpytLieFsv22",
        salt="$2b$12$OazL5oj8/lxxOV5a5j2Nme",
        private_key="4de2d41486ceaffdf0c1778e50cea00000d6549ffe808fa860ecd4e91d9ee1b1",
        role=1,
    )
    database.session.add(new_user)
    new_user = create_user(*user4)
    database.session.add(new_user)
    new_user = create_user(*user3)
    database.session.add(new_user)

    database.session.commit()

    assert database.session.query(User).get(3).role == 3

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"role": 1}
    result = client.put(
        "/users/3/role",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "You can't change it to Owner role!"


def test_put_user_role_missing_role(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"role": 2}
    result = client.put(
        "/users/2/role",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 404
    assert result.get_json()["error"] == "Role ID not found!"


def test_put_other_user_role_missing_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"role": 2}
    result = client.put(
        "/users/2/role",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 404
    assert result.get_json()["error"] == "User not found!"


# PUT USER PASSWORD


def test_put_other_user_password_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
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
    headers = {
        "token": token.decode("UTF-8"),
    }
    new_password = "BrandNewPassword123"
    payload = {"password": new_password}
    result = client.put(
        "/users/2/password",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 200
    assert result.get_json() == {"message": "User updated successfully!"}
    assert checkpw(
        new_password.encode("UTF-8"),
        user.salt.encode("UTF-8") + user.hashed_password.encode("UTF-8"),
    )


def test_put_user_password_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    headers = {}
    new_password = "BrandNewPassword123"
    payload = {"password": new_password}
    result = client.put(
        "/users/2/password",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_put_user_password_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "secretitis")
    headers = {
        "token": token.decode("UTF-8"),
    }
    new_password = "BrandNewPassword123"
    payload = {"password": new_password}
    result = client.put(
        "/users/2/password",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_put_other_user_password_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    new_password = "BrandNewPassword123"
    payload = {"password": new_password}
    result = client.put(
        "/users/1/password",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "You're not allowed to change other user data!"


def test_put_own_user_password_success(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = User(
        email="owner@owner.com",
        hashed_password="RcEEa25p/APCVGFaBaiZpytLieFsv22",
        salt="$2b$12$OazL5oj8/lxxOV5a5j2Nme",
        private_key="4de2d41486ceaffdf0c1778e50cea00000d6549ffe808fa860ecd4e91d9ee1b1",
        role=1,
    )
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
    headers = {
        "token": token.decode("UTF-8"),
    }
    new_password = "BrandNewPassword123"
    payload = {"password": new_password}
    result = client.put(
        "/users/3/password",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 200
    assert result.get_json() == {"message": "User updated successfully!"}
    assert checkpw(
        new_password.encode("UTF-8"),
        user.salt.encode("UTF-8") + user.hashed_password.encode("UTF-8"),
    )


def test_put_other_user_email_missing_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    new_password = "BrandNewPassword123"
    payload = {"password": new_password}
    result = client.put(
        "/users/2/password",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 404
    assert result.get_json()["error"] == "User not found!"


# DELETE USER


def test_delete_other_user_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
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
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/users/2", headers=headers, content_type="application/json")

    assert result.status_code == 200
    assert result.get_json() == {"message": "User deleted successfully!"}
    assert database.session.query(User).get(2) is None


def test_delete_user_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    headers = {}
    result = client.delete("/users/2", headers=headers, content_type="application/json")

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_delete_user_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "secretitis")
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/users/2", headers=headers, content_type="application/json")

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_delete_other_user_unauthorized(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 2}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/users/1", headers=headers, content_type="application/json")

    assert result.status_code == 403
    assert (
        result.get_json()["error"]
        == "You're not allowed to delete this user information!"
    )


def test_delete_own_user(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = User(
        email="owner@owner.com",
        hashed_password="RcEEa25p/APCVGFaBaiZpytLieFsv22",
        salt="$2b$12$OazL5oj8/lxxOV5a5j2Nme",
        private_key="4de2d41486ceaffdf0c1778e50cea00000d6549ffe808fa860ecd4e91d9ee1b1",
        role=1,
    )
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
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/users/3", headers=headers, content_type="application/json")
    user_groups = database.session.query(UserGroup).filter_by(user=3).all()

    assert result.status_code == 403
    assert (
        result.get_json()["error"]
        == "You're not allowed to delete this user information!"
    )
    assert database.session.query(User).get(3) is not None


def test_delete_other_user_missing_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = User(
        email="tech@gibberish.com",
        hashed_password="2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
        salt="$2b$12$tufn64/0gSIAdprqBrRzC.",
        private_key="fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        role=1,
    )
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    result = client.delete("/users/2", headers=headers, content_type="application/json")

    assert result.status_code == 404
    assert result.get_json()["error"] == "User not found!"


# SEARCH USERS
# TODO: Update search unit tests


def test_search_users_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = User(
        email="tech@gibberish.com",
        hashed_password="2amt5MXKdLhEEL8FiQLcl8Mp0FNhZI6",
        salt="$2b$12$tufn64/0gSIAdprqBrRzC.",
        private_key="fd062d885b24bda173f6aa534a3418bcafadccecfefe2f8c6f5a8db563549ced",
        role=1,
    )
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
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"email": "anemail@anemail.com"}
    result = client.post(
        "/users/search",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 200
    assert len(result.get_json()) == 1
    assert result.get_json()[0]["id"] == 2


def test_search_users_nomatch(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = User(
        email="owner@owner.com",
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
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"role": 8}
    result = client.post(
        "/users/search",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 200
    assert len(result.get_json()) == 0


def test_search_users_onematch(client, database, cleanup):
    new_role = create_role(*owner_role)
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
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"role": 3, "group": 1, "email": "tech@gibberish.com"}
    result = client.post(
        "/users/search",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 200
    assert len(result.get_json()) == 1
    assert result.get_json()[0]["id"] == 2


def test_search_users_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    headers = {}
    payload = {"email": "anemail@anemail.com"}
    result = client.post(
        "/users/search",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_search_users_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*user_role)
    database.session.add(new_role)
    new_user = create_user(*user1)
    database.session.add(new_user)
    new_user = create_user(*user2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "secretitis")
    headers = {
        "token": token.decode("UTF-8"),
    }
    payload = {"email": "anemail@anemail.com"}
    result = client.post(
        "/users/search",
        headers=headers,
        data=dumps(payload),
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"
