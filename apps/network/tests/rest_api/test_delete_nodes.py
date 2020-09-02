from unittest.mock import patch


def test_delete_grid_node_bad_request_json(client):
    """assert that the endpoint returns a 400 for malformed JSON."""
    result = client.delete("/delete-node", data="{bad", content_type="application/json")
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


def test_delete_grid_node_bad_request_key_error(client):
    """assert that the endpoint returns a 400 for missing keys in sent JSON."""
    result = client.delete("/delete-node", json={"test": "data"})
    assert result.status_code == 400
    assert result.get_json().get("message") == ("Invalid JSON format.")


@patch("src.app.routes.network.network_manager")
def test_delete_grid_node_not_registered(mock_network_manager, client):
    """assert that the endpoint returns a 400 for a node not yet registered."""
    mock_network_manager.delete_node.return_value = False

    result = client.delete(
        "/delete-node", json={"node-id": "data", "node-address": "data"}
    )

    assert result.status_code == 409
    assert (
        result.get_json().get("message") == "This ID was not found in connected nodes"
    )


@patch("src.app.routes.network.network_manager")
def test_delete_grid_node_internal_server_error(mock_network_manager, client):
    """assert that the endpoint returns a 500 for an unknown error
    condition."""
    mock_network_manager.delete_node.side_effect = RuntimeError("test")

    result = client.delete(
        "/delete-node", json={"node-id": "data", "node-address": "data"}
    )

    assert result.status_code == 500
    assert result.get_json().get("message") == "test"
