def test_create_group(client):
    result = client.post(
        "/groups/",
        data={"group_name": "group test", "members": ["239y94asd", "whor244123"]},
    )
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Group created succesfully!"}


def test_get_all_groups(client):
    result = client.get("/groups/")
    assert result.status_code == 200
    assert result.get_json() == {"groups": ["GroupA", "GroupB", "GroupC"]}


def test_get_specific_group(client):
    result = client.get("/groups/5484626")
    assert result.status_code == 200
    assert result.get_json() == {"group": {"name": "Group A", "id": "5484626"}}


def test_update_group(client):
    result = client.put("/groups/546313", data={"group_configs": "{configs}"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Group was updated succesfully!"}


def test_delete_group(client):
    result = client.delete("/groups/546313")
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Group was deleted succesfully!"}
