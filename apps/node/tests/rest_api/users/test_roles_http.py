from json import loads

import jwt
import pytest
from flask import current_app as app

from src.app.main.core.exceptions import PyGridError
from src.app.main.database import Role, User, create_role, create_user, model_to_json

payload = (
    '{"name": "mario mario", "can_triage_requests": false,'
    '"can_edit_settings": false, "can_create_users": true,'
    '"can_create_groups": true, "can_edit_roles": false,'
    '"can_manage_infrastructure": false, "can_upload_data": false}'
)
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

    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
    }
    result = client.post(
        "/roles", data=payload, content_type="application/json", headers=headers
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_post_role_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {"token": token.decode("UTF-8")}
    result = client.post(
        "/roles", data=payload, content_type="application/json", headers=headers
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_post_role_bad_data(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }

    result = client.post(
        "/roles", data="{bad", headers=headers, content_type="application/json"
    )
    assert result.status_code == 400
    assert result.get_json()["error"] == JSON_DECODE_ERR_MSG


def test_post_role_invalid_key(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {"private-key": "IdoNotExist", "token": token.decode("UTF-8")}
    result = client.post(
        "/roles", data=payload, content_type="application/json", headers=headers
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_post_role_invalid_token(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"asdsadad": 124356}, app.config["SECRET_KEY"])
    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.post(
        "/roles", data=payload, content_type="application/json", headers=headers
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_post_role_user_with_missing_role(client, database, cleanup):

    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private-key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.post(
        "/roles", data=payload, content_type="application/json", headers=headers
    )

    assert result.status_code == 404
    assert result.get_json()["error"] == "Role ID not found!"


def test_post_role_missing_user(client, database, cleanup):
    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private-key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.post(
        "/roles", data=payload, content_type="application/json", headers=headers
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


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
        "/roles", data=payload, content_type="application/json", headers=headers
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "User is not authorized for this operation!"


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
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.post(
        "/roles", data=payload, content_type="application/json", headers=headers
    )

    expected_role = loads(payload)
    expected_role["id"] = 3  # Two roles already inserted

    assert result.status_code == 200
    assert result.get_json()["role"] == expected_role


# GET ALL ROLES


def test_get_all_roles_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb"
    }
    result = client.get(
        "/roles", data=payload, headers=headers, content_type="application/json"
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_get_all_roles_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {"token": token.decode("UTF-8")}
    result = client.get(
        "/roles", data=payload, headers=headers, content_type="application/json"
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_get_all_roles_invalid_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private_key": "siohfigdadANDVBSIAWE0WI21Y8OR1082ORHFEDNSLCSADIJOKA",
        "token": token.decode("UTF-8"),
    }
    result = client.get(
        "/roles", data=payload, headers=headers, content_type="application/json"
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


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
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.get(
        "/roles", data=payload, headers=headers, content_type="application/json"
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


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
    result = client.get("/roles", content_type="application/json", headers=headers)

    expected_roles = [model_to_json(role1), model_to_json(role2)]

    assert result.status_code == 200
    assert result.get_json()["roles"] == expected_roles


# GET SINGLE ROLE


def test_get_role_missing_key(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {"token": token.decode("UTF-8")}

    result = client.get(
        "/roles/1", data=payload, headers=headers, content_type="application/json"
    )
    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_get_role_missing_token(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
    }
    result = client.get(
        "/roles/1", data=payload, headers=headers, content_type="application/json"
    )

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_get_role_invalid_key(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {"private_key": "IdoNotExist", "token": token.decode("UTF-8")}
    result = client.get(
        "/roles/1", data=payload, headers=headers, content_type="application/json"
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_get_role_invalid_token(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"asdsadad": 124356}, app.config["SECRET_KEY"])
    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.get(
        "/roles/1", data=payload, headers=headers, content_type="application/json"
    )

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_get_role_missing_user(client):
    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.get("/roles/2", content_type="application/json", headers=headers)
    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_get_role_missing_role(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
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
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.get("/roles/1", content_type="application/json", headers=headers)

    assert result.status_code == 403
    assert result.get_json()["error"] == "User is not authorized for this operation!"


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
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.get("/roles/1", content_type="application/json", headers=headers)

    expected_role = model_to_json(role1)

    assert result.status_code == 200
    assert result.get_json()["role"] == expected_role


# PUT ROLE


def test_put_role_missing_key(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {"token": token.decode("UTF-8")}
    result = client.put(
        "/roles/1", data=payload, headers=headers, content_type="application/json"
    )
    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_put_role_missing_token(client, database, cleanup):
    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb"
    }
    result = client.put(
        "/roles/1", data=payload, headers=headers, content_type="application/json"
    )
    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_put_role_invalid_key(client, database, cleanup):
    new_role = create_role(*owner_role)

    new_user = create_user(*user_1)

    database.session.add(new_role)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private_key": "dsapksasdp12-04290u83t5r752tyvdwhbsacnxz",
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/1", data=payload, headers=headers, content_type="application/json"
    )
    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_put_role_invalid_token(client, database, cleanup):
    new_role = create_role(*owner_role)

    new_user = create_user(*user_1)

    database.session.add(new_role)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, "1029382trytdfsvcbxz")
    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/1", data=payload, headers=headers, content_type="application/json"
    )
    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_put_role_bad_data(client, database, cleanup):
    new_role = create_role(*owner_role)

    new_user = create_user(*user_1)

    database.session.add(new_role)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
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
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/1", data=payload, content_type="application/json", headers=headers
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
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/1", data=payload, content_type="application/json", headers=headers
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
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/3", data=payload, content_type="application/json", headers=headers
    )

    assert result.status_code == 404
    assert result.get_json()["error"] == "Role ID not found!"


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
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.put(
        "/roles/1", data=payload, content_type="application/json", headers=headers
    )

    expected_role = loads(payload)
    expected_role["id"] = 1

    assert result.status_code == 200
    assert result.get_json()["role"] == expected_role


# DELETE ROLE


def test_delete_role_missing_key(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user_1)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {"token": token.decode("UTF-8")}
    result = client.delete("/roles/2", content_type="application/json", headers=headers)

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_delete_role_missing_token(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user_1)
    database.session.add(new_user)
    database.session.commit()

    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb"
    }
    result = client.delete("/roles/2", content_type="application/json", headers=headers)

    assert result.status_code == 400
    assert result.get_json()["error"] == "Missing request key!"


def test_delete_role_invalid_key(client, database, cleanup):
    new_role = create_role(*owner_role)
    database.session.add(new_role)
    new_role = create_role(*admin_role)
    database.session.add(new_role)
    new_user = create_user(*user_1)
    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private_key": "1230896843rtfsvdjb123453212098792171766n",
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/roles/2", content_type="application/json", headers=headers)

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


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
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/roles/2", content_type="application/json", headers=headers)

    assert result.status_code == 403
    assert result.get_json()["error"] == "Invalid credentials!"


def test_delete_role_missing_role(client, database, cleanup):

    new_user = create_user(*user_1)

    database.session.add(new_user)
    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
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
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
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
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/roles/3", headers=headers, content_type="application/json")

    assert result.status_code == 404
    assert result.get_json()["error"] == "Role ID not found!"


def test_delete_role_success(client, database, cleanup):
    new_role = create_role(*admin_role)
    database.session.add(new_role)

    new_role = create_role(*owner_role)
    database.session.add(new_role)

    new_user = create_user(*user_2)
    database.session.add(new_user)

    database.session.commit()

    token = jwt.encode({"id": 1}, app.config["SECRET_KEY"])
    headers = {
        "private_key": "3c777d6e1cece1e78aa9c26ae7fa2ecf33a6d3fb1db7c1313e7b79ef3ee884eb",
        "token": token.decode("UTF-8"),
    }
    result = client.delete("/roles/1", headers=headers, content_type="application/json")

    assert result.status_code == 200
    assert database.session.query(Role).get(1) is None
