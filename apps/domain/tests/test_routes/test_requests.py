def test_create_user(client):
    result = client.post(
        "/dcfl/requests",
        data={"id": "61612325", "dataset": "12354", "reason": " reason sample"},
    )
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Request created succesfully!"}


def test_get_all_requests(client):
    result = client.get("/dcfl/requests")
    assert result.status_code == 200
    assert result.get_json() == {
        "requests": [
            {"id": "35654sad6ada", "reason": "request A reason"},
            {"id": "adfarf3f1af5", "reason": "request B reason"},
            {"id": "fas4e6e1fas", "reason": "request C reason"},
        ]
    }


def test_get_specific_request(client):
    result = client.get("/dcfl/requests/6516513")
    assert result.status_code == 200
    assert result.get_json() == {
        "request": {"id": "6516513", "reason": "request reason"}
    }


def test_update_request(client):
    result = client.put("/dcfl/requests/546313", data={"request": "{new_request}"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Request updated succesfully!"}


def test_delete_request(client):
    result = client.delete("/dcfl/requests/546313")
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Request deleted succesfully!"}
