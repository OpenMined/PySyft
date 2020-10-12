


def test_create_network(client):
    result = client.post("/networks/", data={"name": "test_network"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Network created succesfully!"}

def test_get_all_network(client):
    result = client.get("/networks/")
    assert result.status_code == 200
    assert result.get_json() == {"networks": ["Net-Gama", "Net-Beta", "Net-Pi"]}


def test_get_specific_network(client):
    result = client.get("/networks/464615")
    assert result.status_code == 200
    assert result.get_json() == {"network": {"name": "Net-Gama", "id": "464615"}}


def test_update_network(client):
    result = client.put("/networks/546313", data={"node": "{new_node}"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Network was updated succesfully!"}

def test_delete_network(client):
    result = client.delete("/networks/546313")
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Network was deleted succesfully!"}


def test_create_autoscaling(client):
    result = client.post("/networks/autoscaling", data={"configs": "{auto-scaling_configs}"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Network auto-scaling created succesfully!"}

def test_get_all_autoscaling_conditions(client):
    result = client.get("/networks/autoscaling/")
    assert result.status_code == 200
    assert result.get_json() ==  {"auto-scalings": ["Condition 1", "Condition 2", "Condition 3"]}

def test_get_specific_autoscaling_condition(client):
    result = client.get("/networks/autoscaling/6413568")
    assert result.status_code == 200
    assert result.get_json() == {"network": {"name": "Net-Gama", "id": "6413568"}}

def test_update_autoscaling_condition(client):
    result = client.put("/networks/autoscaling/6413568", data={"autoscaling": "{new_autoscaling_condition}"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Network auto-scaling was updated succesfully!"}

def test_delete_autoscaling_condition(client):
    result = client.delete("/networks/autoscaling/6413568")
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Network auto-scaling was deleted succesfully!"}
