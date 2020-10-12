

def test_create_tensor(client):
    result = client.post("/dcfl/tensors", data={"tensor": "{serialized_tensor}"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "tensor created succesfully!"}

def test_get_all_tensors(client):
    result = client.get("/dcfl/tensors")
    assert result.status_code == 200
    assert result.get_json() == {
        "tensors": [
            {
                "id": "35654sad6ada",
                "tags": ["tensor-a"],
                "description": "tensor sample",
            },
            {
                "id": "adfarf3f1af5",
                "tags": ["tensor-b"],
                "description": "tensor sample",
            },
            {
                "id": "fas4e6e1fas",
                "tags": ["tensor-c"],
                "description": "tensor sample",
            },
        ]
    }

def test_get_specific_tensor(client):
    result = client.get("/dcfl/tensors/5484626")
    assert result.status_code == 200
    assert result.get_json() == {
        "tensor": {
            "id": "5484626",
            "tags": ["tensor-a"],
            "description": "tensor sample",
        }
    }


def test_update_tensor(client):
    result = client.put("/dcfl/tensors/546313", data={"tensor": "new_serialized_tensor"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "tensor changed succesfully!"}

def test_delete_tensor(client):
    result = client.delete("/dcfl/tensors/546313")
    assert result.status_code == 200
    assert result.get_json() == {"msg": "tensor deleted succesfully!"}
