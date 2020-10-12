

def test_create_dataset(client):
    result = client.post("/dcfl/datasets", data={"dataset": "{serialized_dataset}"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Dataset created succesfully!"}

def test_get_all_datasets(client):
    result = client.get("/dcfl/datasets")
    assert result.status_code == 200
    assert result.get_json() == {
        "datasets": [
            {
                "id": "35654sad6ada",
                "tags": ["dataset-a"],
                "description": "Dataset sample",
            },
            {
                "id": "adfarf3f1af5",
                "tags": ["dataset-b"],
                "description": "Dataset sample",
            },
            {
                "id": "fas4e6e1fas",
                "tags": ["dataset-c"],
                "description": "Dataset sample",
            },
        ]
    }

def test_get_specific_dataset(client):
    result = client.get("/dcfl/datasets/5484626")
    assert result.status_code == 200
    assert result.get_json() == {
        "dataset": {
            "id": "5484626",
            "tags": ["dataset-a"],
            "description": "Dataset sample",
        }
    }


def test_update_dataset(client):
    result = client.put("/dcfl/datasets/546313", data={"dataset": "new_serialized_dataset"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Dataset changed succesfully!"}

def test_delete_dataset(client):
    result = client.delete("/dcfl/datasets/546313")
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Dataset deleted succesfully!"}
