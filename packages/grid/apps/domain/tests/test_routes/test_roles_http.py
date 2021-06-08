# stdlib
from json import dumps
from json import loads

# third party
from flask import current_app as app
import jwt
import pytest
from src.main.core.database import Role
from src.main.core.database import User
from src.main.core.database import create_role
from src.main.core.database import create_user
from src.main.core.database import model_to_json

payload = {
    "name": "mario mario",
    "can_triage_requests": False,
    "can_edit_settings": False,
    "can_create_users": True,
    "can_create_groups": True,
    "can_edit_roles": False,
    "can_manage_infrastructure": False,
    "can_upload_data": False,
}

JSON_DECODE_ERR_MSG = (
    "Expecting property name enclosed in " "double quotes: line 1 column 2 (char 1)"
)
owner_role = ("Owner", True, True, True, True, True, True, True)
admin_role = ("Administrator", True, True, True, True, False, False, True)
user_role = ("User", False, False, False, False, False, False, False)
officer_role = ("Compliance Officer", True, False, False, False, False, False, False)
user_1 = (
    "tech@gibberish.com",
    "BDEB6E8EE39B6C70835993486C9E65DC",
    "]GBF[R>GX[9Cmk@DthFT!mhloUc%[f",
    "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
    1,
)
user_2 = (
    "tech@gibberish.com",
    "BDEB6E8EE39B6C70835993486C9E65DC",
    "]GBF[R>GX[9Cmk@DthFT!mhloUc%[f",
    "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
    2,
)


@pytest.fixture
def cleanup(database):
    yield
    try:
        database.session.query(User).delete()
        database.session.query(Role).delete()
        database.session.commit()
    except:
        database.session.rollback()


# POST ROLE


def test_post_role_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    headers = {}
    result = client.post(
        "/roles", data=dumps(payload), content_type="application/json", headers=headers
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_post_role_bad_data(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }

    result = client.post(
        "/roles", data="{bad", headers=headers, content_type="application/json"
    )
    assert result.status_code == 400


def test_post_role_invalid_token(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"asdsadad": 124356}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.post(
        "/roles", data=dumps(payload), content_type="application/json", headers=headers
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_post_role_missing_user(client, database, cleanup):
    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private-key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.post(
        "/roles", data=dumps(payload), content_type="application/json", headers=headers
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


"""
def test_post_role_unauthorized_user(client, database, cleanup):
    new_role = create_role(*admin_role)

    new_user = create_user(*user_1)

    database.session.add(new_role)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private-key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.post(
        "/roles", data=dumps(payload), content_type="application/json", headers=headers
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "User is not authorized for this operation!"
"""


def test_post_role_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.post(
        "/roles", data=dumps(payload), content_type="application/json", headers=headers
    )

    expected_role = payload.copy()
    expected_role["id"] = 3  # Two roles already inserted

    assert result.status_code == 204


# GET ALL ROLES


def test_get_all_roles_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    headers = {}
    result = client.get(
        "/roles", data=dumps(payload), headers=headers, content_type="application/json"
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_get_all_roles_invalid_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, "totally a secret, trust me")
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get(
        "/roles", data=dumps(payload), headers=headers, content_type="application/json"
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


"""
def test_get_all_roles_user_with_missing_role(client, database, cleanup):

    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private-key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.get("/roles", content_type="application/json", headers=headers)

    assert result.status_code == 404
    assert result.get_json()["error"] == "Role ID not found!"

def test_get_all_roles_unauthorized_user(client, database, cleanup):
    new_role = create_role(*user_role)

    new_user = create_user(*user_1)

    database.session.add(new_role)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private-key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.get("/roles", content_type="application/json", headers=headers)

    assert result.status_code == 403
    assert result.get_json()["error"] == "User is not authorized for this operation!"
"""


def test_get_all_roles_success(client, database, cleanup):
    role1 = create_role(*user_role)
    database.session.add(role1)

    role2 = create_role(*admin_role)
    database.session.add(role2)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private-key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.get("/roles", headers=headers)

    assert result.status_code == 200
    assert result.get_json() == [
        {
            "can_create_groups": False,
            "can_create_users": False,
            "can_edit_roles": False,
            "can_edit_settings": False,
            "can_manage_infrastructure": False,
            "can_triage_requests": False,
            "can_upload_data": False,
            "id": 1,
            "name": "User",
        },
        {
            "can_create_groups": True,
            "can_create_users": True,
            "can_edit_roles": False,
            "can_edit_settings": True,
            "can_manage_infrastructure": False,
            "can_triage_requests": True,
            "can_upload_data": True,
            "id": 2,
            "name": "Administrator",
        },
    ]


# GET SINGLE ROLE


def test_get_role_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    headers = {}
    result = client.get(
        "/roles/1",
        data=dumps(payload),
        headers=headers,
        content_type="application/json",
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_get_role_invalid_token(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"asdsadad": 124356}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get(
        "/roles/1",
        data=dumps(payload),
        headers=headers,
        content_type="application/json",
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_get_role_missing_user(client, database, cleanup):
    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get("/roles/2", content_type="application/json", headers=headers)
    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


"""
def test_get_role_missing_role(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get("/roles/1", content_type="application/json", headers=headers)

    assert result.status_code == 404
    assert result.get_json()["error"] == "Role ID not found!"

def test_get_role_unauthorized_user(client, database, cleanup):
    new_role = create_role(*user_role)

    new_user = create_user(*user_1)

    database.session.add(new_role)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get("/roles/1", content_type="application/json", headers=headers)

    assert result.status_code == 403
    assert result.get_json()["error"] == "User is not authorized for this operation!"
"""


def test_get_role_success(client, database, cleanup):
    role1 = create_role(*user_role)
    database.session.add(role1)

    role2 = create_role(*admin_role)
    database.session.add(role2)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.get("/roles/1", headers=headers)

    assert result.status_code == 200
    assert result.get_json() == {
        "id": 1,
        "name": "User",
        "can_triage_requests": False,
        "can_edit_settings": False,
        "can_create_users": False,
        "can_create_groups": False,
        "can_edit_roles": False,
        "can_manage_infrastructure": False,
        "can_upload_data": False,
    }


# PUT ROLE


def test_put_role_missing_token(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    headers = {}
    result = client.put(
        "/roles/1",
        data=dumps(payload),
        headers=headers,
        content_type="application/json",
    )
    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_put_role_invalid_token(client, database, cleanup):
    new_role = create_role(*owner_role)

    new_user = create_user(*user_1)

    database.session.add(new_role)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, "1029382trytdfsvcbxz")
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/1",
        data=dumps(payload),
        headers=headers,
        content_type="application/json",
    )
    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


"""
def test_put_role_bad_data(client, database, cleanup):
    new_role = create_role(*owner_role)

    new_user = create_user(*user_1)

    database.session.add(new_role)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/1", data="{bad", headers=headers, content_type="application/json"
    )
    assert result.status_code == 400
    assert result.get_json()["error"] == JSON_DECODE_ERR_MSG


def test_put_role_user_with_missing_role(client, database, cleanup):

    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/1",
        data=dumps(payload),
        content_type="application/json",
        headers=headers,
    )

    assert result.status_code == 404
    assert result.get_json()["error"] == "Role ID not found!"

def test_put_role_unauthorized_user(client, database, cleanup):
    new_role = create_role(*admin_role)

    new_user = create_user(*user_1)

    database.session.add(new_role)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/1",
        data=dumps(payload),
        content_type="application/json",
        headers=headers,
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "User is not authorized for this operation!"

def test_put_over_missing_role(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)

    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/3",
        data=dumps(payload),
        content_type="application/json",
        headers=headers,
    )

    assert result.status_code == 404
    assert result.get_json()["error"] == "Role ID not found!"
"""


def test_put_role_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)

    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/1",
        data=dumps(payload),
        content_type="application/json",
        headers=headers,
    )

    assert result.status_code == 204


# DELETE ROLE


def test_delete_role_missing_token(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user_1)
    database.session.add(new_user)
    database.session.commit()

    headers = {}
    result = client.delete("/roles/2", content_type="application/json", headers=headers)

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_delete_role_invalid_token(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user_1)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, "213p4u4trgsvczxnwdaere67yiukyhj")
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/roles/2", content_type="application/json", headers=headers)

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


"""

def test_delete_role_missing_role(client, database, cleanup):

    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/roles/1", headers=headers, content_type="application/json")

    assert result.status_code == 404
    assert result.get_json()["error"] == "Role ID not found!"


def test_delete_role_unauthorized_user(client, database, cleanup):
    new_role = create_role(*admin_role)
    new_user = create_user(*user_1)
    database.session.add(new_role)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/roles/1", headers=headers, content_type="application/json")

    assert result.status_code == 403
    assert result.get_json()["error"] == "User is not authorized for this operation!"


def test_delete_role_user_with_missing_role(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)

    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/roles/3", headers=headers, content_type="application/json")

    assert result.status_code == 404
    assert result.get_json()["error"] == "Role ID not found!"
"""


def test_delete_role_success(client, database, cleanup):
    role2 = create_role(*owner_role)
    database.session.add(role2)

    role1 = create_role(*user_role)
    database.session.add(role1)

    new_user = create_user(*user_1)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private-key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/roles/2", headers=headers)

    assert result.status_code == 204
