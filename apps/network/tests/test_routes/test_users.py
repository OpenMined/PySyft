def test_create_user(client):
    result = client.post("/users/", data={"username": "test", "password": "1234"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "User created succesfully!"}


def test_get_all_users(client):
    result = client.get("/users/")
    assert result.status_code == 200
    assert result.get_json() == {"users": ["Bob", "Alice", "James"]}


def test_get_specific_user(client):
    result = client.get("/users/5484626")
    assert result.status_code == 200
    assert result.get_json() == {"user": {"name": "Bob", "id": "5484626"}}


def test_search_users(client):
    result = client.post("/users/search", data={"query": "query_sample"})
    assert result.status_code == 200
    assert result.get_json() == {"users": ["Bob", "Alice", "James"]}


def test_search_users(client):
    result = client.put("/users/546313/email", data={"email": "new_email@email.com"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "User email was changed succesfully!"}


def test_change_password(client):
    result = client.put("/users/546313/password", data={"password": "new_password123"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "User password was changed succesfully!"}


def test_change_role(client):
    result = client.put("/users/546313/role", data={"role": "new_role"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "User role was changed succesfully!"}


def test_delete_user(client):
    result = client.delete("/users/546313")
    assert result.status_code == 200
    assert result.get_json() == {"msg": "User was deleted succesfully!"}


def test_login(client):
    result = client.post(
        "/users/login", data={"username": "user", "password": "pwd123"}
    )
    response = result.get_json()
    assert result.status_code == 200
    assert response["key"] is not None
    assert response["metadata"] is not None
    assert response["msg"] == "Successfully logged in!"
