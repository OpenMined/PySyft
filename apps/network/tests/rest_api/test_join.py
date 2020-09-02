from unittest.mock import patch


def test_join_grid_node_bad_request_json(client):
    """assert that the endpoint returns a 400 for malformed JSON."""
    result = client.post("/join", data="{bad", content_type="application/json")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


def test_join_grid_node_bad_request_key_error(client):
    """assert that the endpoint returns a 400 for missing keys in sent JSON."""
    result = client.post("/join", json={"test": "data"})
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("src.app.routes.network.network_manager")
def test_join_grid_node_already_registered(mock_network_manager, client):
    """assert that the endpoint returns a 400 for an already registered
    node."""
    mock_network_manager.register_new_node.return_value = False

    result = client.post("/join", json={"node-id": "data", "node-address": "data"})

    assert result.status_code == 409
    assert result.get_json().get("message") == "This ID has already been registered"


@patch("src.app.routes.network.network_manager")
def test_join_grid_node_internal_server_error(mock_network_manager, client):
    """assert that the endpoint returns a 500 for an unknown error
    condition."""
    mock_network_manager.register_new_node.side_effect = RuntimeError("test")

    result = client.post("/join", json={"node-id": "data", "node-address": "data"})

    assert result.status_code == 500
    assert result.get_json().get("message") == "test"
