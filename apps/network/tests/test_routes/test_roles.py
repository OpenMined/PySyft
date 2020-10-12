def test_create_role(client):
    result = client.post("/roles/", data={"role": "admin"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Role created succesfully!"}


def test_get_all_roles(client):
    result = client.get("/roles/")
    assert result.status_code == 200
    assert result.get_json() == {"roles": ["RoleA", "RoleB", "RoleC"]}

def test_get_specific_role(client):
    result = client.get("/roles/654816")
    assert result.status_code == 200
    assert result.get_json() == {"role": {"name": "Role A", "id": "654816"}}

def test_update_role(client):
    result = client.put("/roles/654816", data={"role": "{new_role_configs}"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Role was updated succesfully!"}

def test_delete_role(client):
    result = client.delete("/roles/654816")
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Role was deleted succesfully!"}
