
def test_send_association_request(client):
    result = client.post("/association-requests/request", data={"id": "54623156", "address": "159.15.223.162"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Association request sent!"}

def test_receive_association_request(client):
    result = client.post("/association-requests/receive", data={"id": "54623156", "address": "159.15.223.162"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Association request received!"}

def test_reply_association_request(client):
    result = client.post("/association-requests/respond", data={"id": "54623156", "address": "159.15.223.162"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Association request was replied!"}

def get_all_association_requests(client):
    result = client.get("/association-requests/")
    assert result.status_code == 200
    assert result.get_json() == {"association-requests": ["Network A", "Network B", "Network C"]}

def get_specific_association_requests(client):
    result = client.get("/association-requests/51613546")
    assert result.status_code == 200
    assert result.get_json() == {
        "association-request": {
            "ID": "51613546",
            "address": "156.89.33.200",
        }
    }

def delete_association_requests(client):
    result = client.get("/association-requests/51661659")
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Association request deleted!"}
