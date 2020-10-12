

def test_initial_setup(client):
    result = client.post("/setup/", data={"setup": "setup_configs_sample"})
    assert result.status_code == 200
    assert result.get_json() == {"msg": "Running initial setup!"}

def test_get_setup(client):
    result = client.get("/setup/")
    assert result.status_code == 200
    assert result.get_json() == {"setup": {}}
